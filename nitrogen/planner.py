"""Plan-conditioning modules for NitroGen.

A frozen (Stage 1) small VLM planner (e.g. Qwen3.5-0.8B) reads past frames and a
plan (synthetic in Stage 1, transcript-derived in Stage 2) and produces last-layer
hidden states. A learned Perceiver-style resampler distills those into K fixed
"plan tokens", which an adapter maps into NitroGen's vision-hidden space. The K
tokens are injected into the VL cross-attention stream alongside the image tokens.

Plan-dropout replaces the plan tokens with a learned null-plan embedding so that:
  * null plan  -> reproduce the streamer's original action (CFG "unconditional")
  * real plan  -> steer the action toward the plan        (CFG "conditional")
This lets us reuse NitroGen.get_action_with_cfg for plan guidance at inference.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class PlannerConfig(BaseModel):
    """Configuration for the plan-conditioning stack."""

    enabled: bool = Field(default=False, description="Whether plan conditioning is active.")
    backbone_name_or_path: str = Field(
        default="Qwen/Qwen3.5-0.8B",
        description="HF model id or local path for the VLM planner backbone.",
    )
    num_plan_tokens: int = Field(default=8, description="K: number of plan tokens produced.")
    num_chunks: int = Field(default=1, description="A: number of action-chunks one plan spans (cross-chunk). When A>1 the resampler emits K*A tokens reshaped to A blocks of K; a per-example cursor a in [0,A) selects block a for that chunk's injection. A=1 is the single-chunk path (unchanged).")
    resampler_layers: int = Field(default=2, description="Number of cross-attention layers in the resampler.")
    resampler_heads: int = Field(default=8, description="Number of attention heads in the resampler.")
    resampler_query_self_attn: bool = Field(default=False, description="If True, the K resampler queries also self-attend to each other in each layer (true Q-former/BLIP-2 style: self-attn -> cross-attn -> FFN), so the K plan tokens can coordinate/divide labor explicitly. Default False keeps the Perceiver-style cross-attn-only resampler (the current best ckpt).")
    adapter_hidden: int = Field(default=2048, description="Hidden width of the adapter MLP.")
    plan_dropout: float = Field(default=0.15, description="Probability of dropping the plan to null during training.")
    null_mode: str = Field(default="learned", description="How a 'null' plan is realized: 'learned' substitutes a learned null embedding for the K plan tokens; 'masked' leaves the tokens but masks the K plan positions out of the DiT cross-attention (exact base behavior).")
    contrastive_weight: float = Field(default=0.0, description="Weight of the supervised-contrastive auxiliary loss on pooled plan tokens (groups by plan_label) to push apart opposite-direction plans. 0 disables.")
    contrastive_temp: float = Field(default=0.1, description="Temperature for the SupCon plan-token loss.")
    contrastive_mode: str = Field(default="mean", description="Representation used by the SupCon plan-token loss: 'mean' pools over the K tokens (order-blind; opposite orderings share a mean and stay collinear); 'flatten' concatenates the K tokens (order-aware: forces order info into position-specific tokens); 'pertoken' applies SupCon independently per query position and averages (strongest per-position discriminability).")
    distill_weight: float = Field(default=0.0, description="Weight of the privileged-info DISTILLATION loss (EXP-045): pull the (base-plan) student plan tokens toward precomputed teacher plan tokens from the action-augmented prompt P+. 0 disables. Combines a per-token MSE (transfer steering) with an InfoNCE/CLIP term (per-chunk positive vs cross-chunk negatives; de-collinearize).")
    distill_temp: float = Field(default=0.1, description="Temperature for the InfoNCE term of the distillation loss.")
    plan_hidden_size: int = Field(default=1024, description="Hidden size of the planner backbone (== NitroGen vision_hidden_size).")
    plan_adaln: bool = Field(default=False, description="EXP-048: in ADDITION to the K cross-attention plan tokens, give the plan GLOBAL authority by adding a zero-init plan offset into the DiT timestep embedding (adaLN/FiLM). The plan then multiplicatively gates every DiT block + the output. Identity at init (zero-init proj) and null-masked -> base-exact; the override fix for the plan being structurally outvoted (8 plan tokens vs ~256 vision tokens). 0/False disables.")
    dit_temb_dim: int = Field(default=0, description="DiT timestep-embedding (inner) dim; set by NitroGen at construction so the plan-adaLN projection can map plan_dim -> temb_dim. 0 = unset/disabled.")
    freeze_backbone: bool = Field(default=True, description="Freeze the VLM backbone (Stage 1).")
    backbone_dtype: str = Field(default="bfloat16", description="Dtype for the (frozen) backbone.")


class PlanResampler(nn.Module):
    """K learned queries cross-attend to VLM hidden states -> (B, K, dim).

    If `query_self_attn` is True, each layer additionally lets the K queries
    self-attend to each other (Q-former/BLIP-2 style) before cross-attending to
    the VLM hidden states, so the K plan tokens can coordinate explicitly rather
    than only specializing via their learned initialisation (Perceiver style).
    """

    def __init__(self, dim: int = 1024, num_queries: int = 8, num_heads: int = 8,
                 num_layers: int = 2, ff_mult: int = 4, query_self_attn: bool = False):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_self_attn = query_self_attn
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            blk = nn.ModuleDict({
                "ln_q": nn.LayerNorm(dim),
                "ln_kv": nn.LayerNorm(dim),
                "attn": nn.MultiheadAttention(dim, num_heads, batch_first=True),
                "ln_ff": nn.LayerNorm(dim),
                "ff": nn.Sequential(
                    nn.Linear(dim, dim * ff_mult), nn.GELU(),
                    nn.Linear(dim * ff_mult, dim),
                ),
            })
            if query_self_attn:
                blk["ln_sa"] = nn.LayerNorm(dim)
                blk["self_attn"] = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.layers.append(blk)

    def forward(self, h: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: (B, L, dim); key_padding_mask: (B, L) with True == ignore/pad.
        bsz = h.shape[0]
        q = self.queries.unsqueeze(0).expand(bsz, -1, -1).contiguous()
        for blk in self.layers:
            if "self_attn" in blk:
                qs = blk["ln_sa"](q)
                sa_out, _ = blk["self_attn"](qs, qs, qs, need_weights=False)
                q = q + sa_out
            qn = blk["ln_q"](q)
            kv = blk["ln_kv"](h)
            attn_out, _ = blk["attn"](
                qn, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
            )
            q = q + attn_out
            q = q + blk["ff"](blk["ln_ff"](q))
        return q  # (B, K, dim)


class PlanAdapter(nn.Module):
    """Map resampled plan tokens into NitroGen's vision-hidden space (dim -> dim)."""

    def __init__(self, dim: int = 1024, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlanEncoder(nn.Module):
    """Wraps a HF VLM and returns last-layer hidden states for a plan.

    Stage 1 uses encode/teacher-forced mode (no generation). The backbone is
    typically frozen, so hidden states can be precomputed/cached outside the loop.
    """

    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        self._loaded = False
        self.backbone = None
        self.tokenizer = None
        self.processor = None

    def load(self):
        """Lazily load the backbone (avoids the heavy import/download at config time)."""
        if self._loaded:
            return
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        dtype = getattr(torch, self.config.backbone_dtype)
        self.backbone = AutoModelForImageTextToText.from_pretrained(
            self.config.backbone_name_or_path, dtype=dtype, low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.backbone_name_or_path)
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.config.backbone_name_or_path)
        except Exception:
            self.processor = None
        if self.config.freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._loaded = True

    @property
    def text_model(self):
        return self.backbone.model if hasattr(self.backbone, "model") else self.backbone

    def encode_text(self, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        """Text-only encode -> (hidden_states (B,L,d), key_padding_mask (B,L))."""
        self.load()
        if next(self.backbone.parameters()).device != torch.device(device):
            self.backbone.to(device)
        enc = self.tokenizer(texts, return_tensors="pt", padding=True).to(device)
        out = self.text_model(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            output_hidden_states=True, use_cache=False,
        )
        h = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else out.last_hidden_state
        key_padding_mask = enc["attention_mask"] == 0
        return h, key_padding_mask

    def forward(self, *args, **kwargs):
        return self.encode_text(*args, **kwargs)


class PlanHead(nn.Module):
    """Resampler + adapter + learned null-plan token, with plan-dropout.

    Cross-chunk (num_chunks A > 1): the resampler emits K*A tokens (A blocks of K).
    A per-example `cursor` in [0,A) gathers block `cursor` -> (B, K, d), so the
    downstream injection path is identical to the single-chunk case (still K tokens).
    A=1 leaves the resampler at K queries and skips the gather (unchanged behavior).
    """

    def __init__(self, config: PlannerConfig):
        super().__init__()
        self.config = config
        dim = config.plan_hidden_size
        self.num_chunks = config.num_chunks
        self.resampler = PlanResampler(
            dim=dim, num_queries=config.num_plan_tokens * config.num_chunks,
            num_heads=config.resampler_heads, num_layers=config.resampler_layers,
            query_self_attn=config.resampler_query_self_attn,
        )
        self.adapter = PlanAdapter(dim=dim, hidden=config.adapter_hidden)
        self.null_plan = nn.Parameter(torch.randn(config.num_plan_tokens, dim) * 0.02)
        # EXP-048 plan-adaLN: map the pooled plan tokens -> a DiT-temb offset (global FiLM
        # authority, in addition to the K cross-attention tokens). Zero-init -> identity at
        # start (base-exact); null rows are masked to 0 in adaln_cond() so null == base.
        self.adaln_proj = None
        if config.plan_adaln and config.dit_temb_dim > 0:
            self.adaln_proj = nn.Linear(dim, config.dit_temb_dim)
            nn.init.zeros_(self.adaln_proj.weight)
            nn.init.zeros_(self.adaln_proj.bias)

    def forward(self, vlm_hidden: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                dropped: Optional[torch.Tensor] = None,
                cursor: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """vlm_hidden: (B,L,d). dropped: (B,) bool -> use null plan for those rows.
        cursor: (B,) long in [0,A) -> select block a (cross-chunk). None -> block 0.

        Returns (plan_tokens (B,K,d), null_tokens (B,K,d)).
        """
        bsz = vlm_hidden.shape[0]
        K = self.config.num_plan_tokens
        plan = self.adapter(self.resampler(vlm_hidden, key_padding_mask))  # (B, K*A, d)
        if self.num_chunks > 1:
            plan = plan.view(bsz, self.num_chunks, K, plan.shape[-1])  # (B, A, K, d)
            if cursor is None:
                cursor = torch.zeros(bsz, dtype=torch.long, device=plan.device)
            cursor = cursor.to(plan.device).long().clamp_(0, self.num_chunks - 1)
            plan = plan[torch.arange(bsz, device=plan.device), cursor]  # (B, K, d)
        null = self.null_plan.unsqueeze(0).expand(bsz, -1, -1)
        if dropped is not None:
            m = dropped.view(bsz, 1, 1).to(plan.dtype)
            plan = (1 - m) * plan + m * null
        return plan, null

    def null_tokens(self, bsz: int) -> torch.Tensor:
        return self.null_plan.unsqueeze(0).expand(bsz, -1, -1)

    def adaln_cond(self, plan_tokens: torch.Tensor,
                   dropped: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """EXP-048: pooled-plan -> DiT-temb offset (B, dit_temb_dim). Returns None if plan-adaLN
        is disabled. Dropped/null rows are forced to 0 so null == base exactly (and so plan-CFG
        amplifies only the real plan's adaLN delta). Zero-init proj -> 0 at start."""
        if self.adaln_proj is None or plan_tokens is None:
            return None
        cond = self.adaln_proj(plan_tokens.mean(dim=1))  # (B, temb_dim)
        if dropped is not None:
            cond = cond * (~dropped).view(-1, 1).to(cond.dtype)
        return cond
