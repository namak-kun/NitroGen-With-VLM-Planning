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

REPO = "/home/t-nagupta/NitroGen"
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
                 siglip: str = "google/siglip2-large-patch16-256", default_cfg: float = 1.0):
        self.device = device; self.K = K; self.H = H; self.default_cfg = default_cfg
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

    def _load(self, path, which):
        sd = torch.load(path, map_location="cpu", weights_only=False)[which]
        mc = self.CC.model_cfg.model_copy(deep=True)
        mc.planner_cfg.enabled = True
        mc.planner_cfg.num_plan_tokens = self.K
        mc.planner_cfg.null_mode = "masked"
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
        self._cfg = scenario.cfg_scale if scenario.cfg_scale else self.default_cfg
        self.steps = 0

    def act(self, obs: Observation) -> np.ndarray:
        chunk = self._sample_chunk(obs.frame, self._plan_text, self._cfg)
        self.steps += 1
        return chunk

    # ---- plan-CFG sampling on a live frame --------------------------------------------
    def _prep(self, frame_rgb, text):
        pv = self.ip([np.asarray(frame_rgb)], return_tensors="pt")["pixel_values"][0].numpy()
        ex = self.tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
        d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(self.device)
             for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
        d["images"] = d["images"].float()
        d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=self.device)
        d["game_ids"] = torch.zeros(1, dtype=torch.long, device=self.device)
        h, kpm = self.cache.get(text if text else ".")
        d["plan_hidden"] = h.unsqueeze(0).to(self.device)
        d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(self.device)
        return d

    @torch.no_grad()
    def _sample_chunk(self, frame_rgb, plan_text, w):
        m = self.m
        d = self._prep(frame_rgb, plan_text)
        H_, A_dim = m.config.action_horizon, m.config.action_dim
        num_steps = m.num_inference_timesteps; dt = 1.0 / num_steps
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
            actions = torch.randn(1, H_, A_dim, device=self.device, dtype=torch.float32)

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
                if w == 1.0:
                    actions = actions + dt * vel(actions, tb, pt_c, vlm_c, pc_c)
                else:
                    v_c = vel(actions, tb, pt_c, vlm_c, pc_c)
                    v_u = vel(actions, tb, pt_u, vlm_u, pc_u)
                    actions = actions + dt * (v_u + w * (v_c - v_u))
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
