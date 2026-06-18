"""Minimal LoRA for the NitroGen DiT cross-attention.

Adds low-rank adapters to the cross-attention projections (to_q/to_k/to_v/to_out)
of the DiT's cross-attn blocks, so plan conditioning gets extra capacity to learn
position-dependent / temporal routing WITHOUT full fine-tuning of the pretrained
policy. The base Linear weights are frozen; only the rank-r A/B matrices train.

Trade-off note: LoRA on cross-attention also affects attention to the IMAGE tokens,
so it does NOT preserve EXACT null-invariance (unlike a fully frozen DiT). But the
drift is low-rank and small, and at init (B=0) the model is identical to base. For
exact null-invariance one would gate LoRA to only the plan-token key positions; that
is a future refinement.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen nn.Linear with a trainable low-rank delta: y = W0 x + b + (B A) x * (alpha/r)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)  # B=0 -> delta=0 at init (identical to base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype)
        return out + self.scaling * delta


_TARGETS = ("to_q", "to_k", "to_v")


def apply_lora_to_dit(dit, rank: int = 8, alpha: float = 16.0,
                      cross_attn_only: bool = True) -> int:
    """Replace cross-attention projection Linears in the DiT with LoRALinear.

    Returns the number of layers wrapped. `cross_attn_only`: only wrap blocks that do
    cross-attention (even-indexed blocks when interleave_self_attention) — i.e. the
    ones that attend to the plan/image tokens.
    """
    n = 0
    interleave = getattr(dit.config, "interleave_self_attention", False)
    for idx, block in enumerate(dit.transformer_blocks):
        is_cross = (not interleave) or (idx % 2 == 0)
        if cross_attn_only and not is_cross:
            continue
        attn = getattr(block, "attn1", None)
        if attn is None:
            continue
        for name in _TARGETS:
            lin = getattr(attn, name, None)
            if isinstance(lin, nn.Linear):
                setattr(attn, name, LoRALinear(lin, rank=rank, alpha=alpha))
                n += 1
        # to_out is a ModuleList [Linear, Dropout]
        to_out = getattr(attn, "to_out", None)
        if to_out is not None and isinstance(to_out[0], nn.Linear):
            to_out[0] = LoRALinear(to_out[0], rank=rank, alpha=alpha)
            n += 1
    return n


def lora_parameters(module):
    """Yield only the LoRA A/B parameters under a module."""
    for name, p in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield p
