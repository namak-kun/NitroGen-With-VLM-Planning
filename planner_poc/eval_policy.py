"""NitroGenPolicy: the closed-loop bridge between a trained plan-conditioned NitroGen
checkpoint and the game-agnostic eval harness (nitrogen.eval). Implements the Policy
interface: reset(scenario) encodes the plan once; act(observation) runs plan classifier-free
guidance on the live game frame and returns an (H,25) action chunk.

Heavy model deps (torch, transformers, the VLM plan encoder) live here in planner_poc, NOT in
nitrogen.eval.core, so the harness + envs stay importable without a GPU.

Reuses the exact plan-CFG denoise from counterfactual_eval (v = v_uncond + w*(v_cond-v_uncond)),
which is the mechanism the whole approach relies on, and the checkpoint auto-detection
(LoRA rank, plan-adaLN) so any EXP-04x student/teacher checkpoint plugs in.

System-2 cadence: the VLM plan is encoded ONCE at reset (a fixed instruction per scenario). A
future extension can re-invoke a live VLM every `replan_every` control steps to regenerate the
plan from recent frames (closed-loop System-2); the hook is in act() (see `self.steps`).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/planner_poc")

import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from transformers import AutoImageProcessor

from nitrogen.cfg import CkptConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.eval.core import Policy, Observation, Scenario


class NitroGenPolicy(Policy):
    def __init__(self, ckpt_path: str, which: str = "model", device: str = "cuda",
                 K: int = 8, H: int = 18, qwen: str = None,
                 siglip: str = "google/siglip2-large-patch16-256", default_cfg: float = 1.0,
                 replan_every: int = 0, n_frames: int = 4, frame_stride: int = 1):
        self.device = device; self.K = K; self.H = H; self.default_cfg = default_cfg
        # Closed-loop System-2: if replan_every>0, every `replan_every` control steps the VLM
        # re-reads the last `n_frames` frames (sampled every `frame_stride` steps) and GENERATES
        # a fresh plan text. replan_every=0 -> fixed plan encoded once at reset (Stage-1 behavior).
        self.replan_every = replan_every; self.n_frames = n_frames; self.frame_stride = frame_stride
        self.mm_mode = False  # EXP-050: when True, _prep computes plan_hidden from frames+text (encode_multimodal)
        self.mm_text_only = False  # EXP-052: when True, encode_multimodal returns text-token hiddens only
        ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
        self.CC = CkptConfig.model_validate(ck["ckpt_config"])
        self.ip = AutoImageProcessor.from_pretrained(siglip)
        self.pl = PlanEncoder(PlannerConfig(backbone_name_or_path=qwen or f"{REPO}/ckpts/qwen35-0.8b"))
        self.pl.load()
        self.cache = PlanHiddenCache(self.pl, device)
        self.tok = NitrogenTokenizer(NitrogenTokenizerConfig(
            training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
        self.m = self._load(ckpt_path, which)
        self._plan_text = ""
        self._cfg = default_cfg
        self.steps = 0
        self._frame_buf = []          # rolling history of recent frames (np.uint8 HxWx3)
        self._base_instruction = ""   # the scenario objective, used to steer plan generation

    def _load(self, path, which):
        sd = torch.load(path, map_location="cpu", weights_only=False)[which]
        mc = self.CC.model_cfg.model_copy(deep=True)
        mc.planner_cfg.enabled = True
        mc.planner_cfg.num_plan_tokens = self.K
        mc.planner_cfg.null_mode = "masked"
        # Infer the VLM backbone hidden size from the checkpoint's resampler (queries last dim) so
        # 0.8B/2B/9B-trained plan heads all reconstruct with the right resampler dim (the adapter
        # projects backbone_hidden -> plan_hidden = DiT dim). Robust to whatever backbone trained it.
        qk = "plan_head.resampler.queries"
        if qk in sd:
            mc.planner_cfg.backbone_hidden_size = int(sd[qk].shape[-1])
            # A>1 cross-chunk: resampler has K*A queries. Infer A so the plan head rebuilds with the
            # right block count (cursor selects block a). queries.shape[0] == num_plan_tokens * A.
            nq = int(sd[qk].shape[0])
            if nq % self.K == 0 and nq // self.K >= 1:
                mc.planner_cfg.num_chunks = nq // self.K
        lk = [k for k in sd if k.endswith(".lora_A")]
        if lk:
            mc.lora_dit_rank = int(sd[lk[0]].shape[0])
        if any(k.endswith("plan_head.adaln_proj.weight") for k in sd):
            mc.planner_cfg.plan_adaln = True
        m = NitroGen(config=mc, game_mapping=None)
        m.load_state_dict(sd, strict=False)
        return m.to(self.device).eval()

    # ---- Policy interface -------------------------------------------------------------
    def reset(self, scenario: Scenario) -> None:
        self._plan_text = scenario.plan or ""
        self._base_instruction = scenario.objective or "What should I do next?"
        self._cfg = scenario.cfg_scale if scenario.cfg_scale else self.default_cfg
        self.steps = 0
        self._frame_buf = []

    def _sampled_history(self):
        """Last n_frames frames, sampled every frame_stride steps, oldest->newest."""
        if not self._frame_buf:
            return []
        sel = self._frame_buf[::-1][::self.frame_stride][: self.n_frames][::-1]
        return sel

    def regenerate_plan(self) -> str:
        """System-2: read recent frames and generate a fresh plan text (closed-loop)."""
        frames = self._sampled_history()
        if not frames:
            return self._plan_text
        instr = self._base_instruction or "What should I do next?"
        plan = self.pl.generate_plan(frames, self.device, instruction=instr,
                                     prev_plan=self._plan_text or None)
        return plan or self._plan_text

    def act(self, obs: Observation) -> np.ndarray:
        self._frame_buf.append(np.asarray(obs.frame))
        if len(self._frame_buf) > self.n_frames * self.frame_stride + 1:
            self._frame_buf.pop(0)
        # System-2 cadence: regenerate the plan from frames every replan_every steps.
        if self.replan_every and (self.steps % self.replan_every == 0):
            self._plan_text = self.regenerate_plan()
        chunk = self._sample_chunk(obs.frame, self._plan_text, self._cfg)
        self.steps += 1
        return chunk

    # ---- plan-CFG sampling on a live frame --------------------------------------------
    def _prep(self, frame_rgb, text, plan_frames=None, plan_hidden_override=None):
        pv = self.ip([np.asarray(frame_rgb)], return_tensors="pt")["pixel_values"][0].numpy()
        ex = self.tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
        d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(self.device)
             for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
        d["images"] = d["images"].float()
        d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=self.device)
        d["game_ids"] = torch.zeros(1, dtype=torch.long, device=self.device)
        if plan_hidden_override is not None:
            # STALENESS TTA: reuse a precomputed (frame-grounded) plan_hidden instead of re-encoding it on the
            # live frame. `cached` mode passes t=0 hiddens (fully stale); `fresh_ema` passes EMA'd hiddens. The
            # DiT still sees the LIVE frame (images above) -- only the PLAN tokens' grounding is overridden.
            h, kpm = plan_hidden_override
            d["plan_hidden"] = h.to(self.device)
            d["plan_key_padding_mask"] = kpm.to(self.device)
        elif self.mm_mode:
            # FRAME-CONDITIONED (EXP-050): the plan hidden is computed from the recent game
            # frame(s) + plan text via the frozen VLM (encode_multimodal), matching how the mm
            # student was trained. plan_frames = recent history (defaults to the current frame).
            frames = plan_frames if plan_frames else [frame_rgb]
            h, kpm = self.pl.encode_multimodal(frames, text if text else ".", self.device,
                                               text_only=self.mm_text_only)
            d["plan_hidden"] = h.to(self.device)
            d["plan_key_padding_mask"] = kpm.to(self.device)
        else:
            h, kpm = self.cache.get(text if text else ".")
            d["plan_hidden"] = h.unsqueeze(0).to(self.device)
            d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(self.device)
        return d

    @torch.no_grad()
    def sample_chunk_token(self, frame_rgb, plan_tokens, w=16.0, seed=None):
        """GOLD-TOKEN inference (env-free counterfactual override): inject EXPLICIT plan tokens
        (1,K,d) — e.g. a teacher token from a training chunk whose action went LEFT — instead of
        encoding plan text. The Cave Story plan TEXT produces weak/washed tokens that can't
        override NitroGen's rightward prior, but a training-derived 'gold' direction token is
        strong enough: with CFG weight w it flips the action against the frame's prior. CFG pairs
        the injected token (cond) vs the null-masked plan positions (uncond) on the SAME frame.
        Multi-seed selection (vary `seed`) picks the most-extreme chunk.
        """
        m = self.m
        if seed is not None:
            torch.manual_seed(seed)
        d = self._prep(frame_rgb, "x", plan_frames=[frame_rgb] if self.mm_mode else None)
        H_, A_dim = m.config.action_horizon, m.config.action_dim
        num_steps = m.num_inference_timesteps; dt = 1.0 / num_steps
        pt = plan_tokens.to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vis = m.encode_images(d["images"])
            cm = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], torch.tensor([False], device=self.device))
            um = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], torch.tensor([True], device=self.device))
            actions = torch.randn(1, H_, A_dim, device=self.device, dtype=torch.float32)

            def vel(acts, tb, ptk, vlm):
                af = m.action_encoder(acts.to(vis.dtype), tb, d["embodiment_id"])
                vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, af,
                                              d["dropped_images"], game_ids=d["game_ids"], plan_tokens=ptk)
                vl = m.vl_self_attention_model(vl, attention_mask=m._additive_key_mask(vlm, vl.dtype))
                mo = m.model(hidden_states=sa, encoder_hidden_states=vl,
                             encoder_attention_mask=vlm, timestep=tb, plan_cond=None)
                return m.action_decoder(mo, d["embodiment_id"])[:, -H_:].float()

            for i in range(num_steps):
                tb = torch.tensor([int((i / num_steps) * m.num_timestep_buckets)], device=self.device)
                v_c = vel(actions, tb, pt, cm)
                v_u = vel(actions, tb, None, um)
                actions = actions + dt * (v_u + w * (v_c - v_u))
        return actions[0].float().cpu().numpy()  # (H, 25)

    @torch.no_grad()
    def _sample_chunk(self, frame_rgb, plan_text, w, plan_frames=None, null=False,
                      noise_sigma=0.0, noise_last_n=4, noise_seed=None, plan_hidden_override=None):
        """Sample an 18-step action chunk via flow-matching Euler integration. noise_sigma>0 makes the
        sampler STOCHASTIC (SDE-style: add sigma*sqrt(dt)*N(0,1) on the LAST noise_last_n steps) for DDPO
        exploration — diverse chunks from the same (frame, plan). 0 = deterministic (default, unchanged).
        plan_hidden_override=(h,kpm): use a precomputed plan_hidden (staleness TTA cached/EMA modes)."""
        m = self.m
        d = self._prep(frame_rgb, plan_text, plan_frames=plan_frames, plan_hidden_override=plan_hidden_override)
        H_, A_dim = m.config.action_horizon, m.config.action_dim
        num_steps = m.num_inference_timesteps; dt = 1.0 / num_steps
        gen = None
        if noise_seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(noise_seed))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vis = m.encode_images(d["images"])
            dc = dict(d); dc["plan_dropped"] = torch.tensor([False], device=self.device)
            du = dict(d); du["plan_dropped"] = torch.tensor([True], device=self.device)
            pt_c, pdp_c = m.compute_plan_tokens(dc)
            pt_u, pdp_u = m.compute_plan_tokens(du)
            vlm_c = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp_c)
            vlm_u = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp_u)
            pc_c = m.plan_head.adaln_cond(pt_c, torch.tensor([False], device=self.device)) if pt_c is not None else None
            pc_u = m.plan_head.adaln_cond(pt_u, torch.tensor([True], device=self.device)) if pt_u is not None else None
            actions = torch.randn(1, H_, A_dim, device=self.device, dtype=torch.float32, generator=gen)

            def vel(acts, tb, pt, vlm, pc):
                af = m.action_encoder(acts.to(vis.dtype), tb, d["embodiment_id"])
                vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, af,
                                              d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
                vl = m.vl_self_attention_model(vl, attention_mask=m._additive_key_mask(vlm, vl.dtype))
                mo = m.model(hidden_states=sa, encoder_hidden_states=vl,
                             encoder_attention_mask=vlm, timestep=tb, plan_cond=pc)
                return m.action_decoder(mo, d["embodiment_id"])[:, -H_:].float()

            for i in range(num_steps):
                tb = torch.tensor([int((i / num_steps) * m.num_timestep_buckets)], device=self.device)
                if null:
                    # NO-PLAN baseline: pure unconditioned (plan-dropped) velocity -> the frozen
                    # DiT's default action on the frame alone, ignoring the plan entirely.
                    actions = actions + dt * vel(actions, tb, pt_u, vlm_u, pc_u)
                elif w == 1.0:
                    actions = actions + dt * vel(actions, tb, pt_c, vlm_c, pc_c)
                else:
                    v_c = vel(actions, tb, pt_c, vlm_c, pc_c)
                    v_u = vel(actions, tb, pt_u, vlm_u, pc_u)
                    actions = actions + dt * (v_u + w * (v_c - v_u))
                if noise_sigma > 0 and i >= num_steps - noise_last_n:
                    actions = actions + noise_sigma * (dt ** 0.5) * torch.randn(
                        actions.shape, device=self.device, dtype=actions.dtype, generator=gen)
        return actions[0].float().cpu().numpy()  # (H, 25)


if __name__ == "__main__":
    # smoke: load the best override model + drive the dummy grid env's selection matrix.
    from nitrogen.eval import EvalProtocol, StatePredicateDetector, Scenario
    from nitrogen.eval.envs.dummy import DummyGridEnv
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student/plan_stage1_2500.pt"
    pol = NitroGenPolicy(ckpt, default_cfg=float(os.environ.get("CFG", "8")))
    print("NitroGenPolicy loaded:", ckpt, "| plan-adaLN:", pol.m.plan_head.adaln_proj is not None)
    # NOTE: the dummy grid is NOT a game NitroGen was trained on -> this only checks the policy
    # produces sane chunks end-to-end in the harness, not that it plays well.
    fr = np.zeros((256, 256, 3), np.uint8)
    ch = pol._sample_chunk(fr, "move left", pol.default_cfg)
    print("chunk shape:", ch.shape, "| left-stick x mean:", float(ch[:, 21].mean()))
