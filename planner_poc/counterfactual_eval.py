"""COUNTERFACTUAL generalization eval (the real OOD-planning test).

All prior steering evals (eval_stage2_dir.py) measure plan-vs-null DELTAS on plans GROUNDED in
the shown actions. The real question (user): if the streamer went RIGHT (and the frame-alone
model also goes right), does commanding "go LEFT" make the model produce a LEFT *absolute*
action -- the plan OVERRIDING the video/frame? That is counterfactual control => env-free OOD
planning.

For a pool of frames we sample the ABSOLUTE left-stick under:
  - null (frame alone)        -> the model's natural / streamer-like behavior a_null
  - each commanded direction d' in {left,right,up,down} (plans from cluster d')
and report:
  (1) STEERING MATRIX: mean absolute stick (x,y) per command (and null). If commanding left
      gives x<0 and right gives x>0 ON THE SAME FRAMES, the plan controls direction.
  (2) OVERRIDE RATE: among frames whose NULL action clearly goes direction g, command the
      OPPOSITE and measure how often the absolute action SIGN FLIPS to the commanded side.
      This is the sharp counterfactual win (plan beats a committed frame prior).
  (3) PLAN-WINS RATE: fraction of (frame, command d') where the absolute action goes d'.
"""
import glob, json, os, sys
import numpy as np
import torch
REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import transformers
if not isinstance(getattr(transformers.SiglipVisionModel, "vision_model", None), property):
    transformers.SiglipVisionModel.vision_model = property(lambda s: s)
from PIL import Image
from transformers import AutoImageProcessor
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.cfg import CkptConfig
from nitrogen.mm_tokenizers import NitrogenTokenizer, NitrogenTokenizerConfig
from nitrogen.planner import PlanEncoder, PlannerConfig
from nitrogen.training.dataset import PlanHiddenCache
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

device = "cuda"; K = 8; H = 18; JLX, JLY = 21, 22
# axis (index, +sign) that each commanded direction wants the stick to move toward
AX = {"left": (JLX, -1), "right": (JLX, +1), "up": (JLY, -1), "down": (JLY, +1)}
OPP = {"left": "right", "right": "left", "up": "down", "down": "up"}
ck = torch.load(f"{REPO}/ckpts/nitrogen/ng.pt", map_location="cpu", weights_only=False)
CC = CkptConfig.model_validate(ck["ckpt_config"])
ip = AutoImageProcessor.from_pretrained("google/siglip2-large-patch16-256")
pl = PlanEncoder(PlannerConfig(backbone_name_or_path=f"{REPO}/ckpts/qwen35-0.8b")); pl.load()
cache = PlanHiddenCache(pl, device)
tok = NitrogenTokenizer(NitrogenTokenizerConfig(training=False, num_plan_tokens=K, action_horizon=H, max_sequence_length=256 + K))
LOOKUP = json.load(open("/tmp/stage2_plan_lookup.json"))
AUG = os.environ.get("AUGMENT_PLANS") == "1"
if AUG:
    from nitrogen.training.actions import summarize_chunk


def load(path, which="model"):
    sd = torch.load(path, map_location="cpu", weights_only=False)[which]
    mc = CC.model_cfg.model_copy(deep=True); mc.planner_cfg.enabled = True
    mc.planner_cfg.num_plan_tokens = K; mc.planner_cfg.null_mode = "masked"
    lk = [k for k in sd if k.endswith(".lora_A")]
    if lk:
        mc.lora_dit_rank = int(sd[lk[0]].shape[0])
    # EXP-048: rebuild the plan-adaLN proj if the checkpoint has it (else adaln_cond -> None).
    if any(k.endswith("plan_head.adaln_proj.weight") for k in sd):
        mc.planner_cfg.plan_adaln = True
    m = NitroGen(config=mc, game_mapping=None)
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not unexp, unexp[:5]
    return m.to(device).eval()


def sample_stick(m, png, text, dropped, seed):
    """Return the mean absolute left-stick (x, y) of the sampled action chunk."""
    fr = np.asarray(Image.open(png).convert("RGB"))
    pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
         for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
    d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else ".")
    d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
    d["plan_dropped"] = torch.tensor([dropped], dtype=torch.bool, device=device)
    torch.manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = m.get_action(d)["action_tensor"][0].float().cpu().numpy()
    return float(a[:, JLX].mean()), float(a[:, JLY].mean())


def _prep(png, text):
    fr = np.asarray(Image.open(png).convert("RGB"))
    pv = ip([fr], return_tensors="pt")["pixel_values"][0].numpy()
    ex = tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
    d = {k: torch.as_tensor(np.asarray(ex[k])).unsqueeze(0).to(device)
         for k in ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]}
    d["images"] = d["images"].float(); d["embodiment_id"] = torch.zeros(1, dtype=torch.long, device=device)
    d["game_ids"] = torch.zeros(1, dtype=torch.long, device=device)
    h, kpm = cache.get(text if text else ".")
    d["plan_hidden"] = h.unsqueeze(0).to(device); d["plan_key_padding_mask"] = kpm.unsqueeze(0).to(device)
    return d


def sample_stick_cfg(m, png, plan_text, w, seed):
    """Plan classifier-free guidance: at each flow step combine the velocity predicted with
    the PLAN tokens (cond) and with the NULL/dropped tokens (uncond):
        v = v_uncond + w*(v_cond - v_uncond)
    w=0 => null (frame alone); w=1 => plain plan; w>1 => amplify the plan delta (push the
    action across the frame prior). Returns mean absolute left-stick (x,y)."""
    d = _prep(png, plan_text)
    H_ = m.config.action_horizon; A_dim = m.config.action_dim
    num_steps = m.num_inference_timesteps; dt = 1.0 / num_steps
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        vis = m.encode_images(d["images"])
        dcond = dict(d); dcond["plan_dropped"] = torch.tensor([False], device=device)
        duncond = dict(d); duncond["plan_dropped"] = torch.tensor([True], device=device)
        pt_c, pdp_c = m.compute_plan_tokens(dcond)
        pt_u, pdp_u = m.compute_plan_tokens(duncond)
        vlm_c = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp_c)
        vlm_u = m.apply_null_mask(d["vl_token_ids"], d["vl_attn_mask"], pdp_u)
        # EXP-048 plan-adaLN offset: real plan for cond, masked-to-0 for uncond (so CFG
        # amplifies the adaLN delta too). adaln_cond returns None if plan-adaLN is disabled.
        pc_c = m.plan_head.adaln_cond(pt_c, torch.tensor([False], device=device)) if pt_c is not None else None
        pc_u = m.plan_head.adaln_cond(pt_u, torch.tensor([True], device=device)) if pt_u is not None else None
        actions = torch.randn(1, H_, A_dim, generator=g, device=device, dtype=torch.float32)

        def vel(acts, tb, pt, vlm, pc):
            af = m.action_encoder(acts.to(vis.dtype), tb, d["embodiment_id"])
            vl, sa = m.prepare_input_embs(d["vl_token_ids"], d["sa_token_ids"], vis, af,
                                          d["dropped_images"], game_ids=d["game_ids"], plan_tokens=pt)
            vl = m.vl_self_attention_model(vl, attention_mask=m._additive_key_mask(vlm, vl.dtype))
            mo = m.model(hidden_states=sa, encoder_hidden_states=vl, encoder_attention_mask=vlm,
                         timestep=tb, plan_cond=pc)
            return m.action_decoder(mo, d["embodiment_id"])[:, -H_:].float()

        for i in range(num_steps):
            tb = torch.tensor([int((i / num_steps) * m.num_timestep_buckets)], device=device)
            v_c = vel(actions, tb, pt_c, vlm_c, pc_c)
            v_u = vel(actions, tb, pt_u, vlm_u, pc_u)
            actions = actions + dt * (v_u + w * (v_c - v_u))
    a = actions[0].float().cpu().numpy()
    return float(a[:, JLX].mean()), float(a[:, JLY].mean())


def build():
    """clusters {dir:[plan]}, and frames [(png, real_dir)]."""
    clusters = {"left": [], "right": [], "up": [], "down": []}
    frames = []
    for md in sorted(glob.glob("/tmp/stage1_big/**/metadata.json", recursive=True)):
        m_ = json.load(open(md)); uuid = m_["uuid"]
        png = f"/tmp/frames_pre/{uuid}.png"
        if uuid not in LOOKUP or not os.path.exists(png):
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 303, H, 2) \
            or assemble_chunk(a["buttons"], a["j_left"], a["j_right"], 3, H, 2)
        if rc is None:
            continue
        dd = chunk_dominant_dir(rc)
        plan = LOOKUP[uuid]["plan"]
        if AUG:
            plan = plan + " To do this I take the following actions: " + summarize_chunk(rc) + "."
        if dd in clusters:
            clusters[dd].append(plan)
        frames.append((png, dd))
    return clusters, frames


def cfg_sweep(m, scales):
    clusters, frames = build()
    rng = np.random.RandomState(0)
    n_plan = 3
    pool_plans = {d: [clusters[d][i] for i in rng.choice(len(clusters[d]), min(n_plan, len(clusters[d])), replace=False)]
                  for d in AX}
    fr_pool = frames[:20]
    print(f"CFG sweep: frames {len(fr_pool)}, plans/dir {n_plan}, scales {scales}")
    print("(w=0 -> frame alone; w=1 -> plain plan; w>1 -> amplify plan delta past the frame prior)\n")
    for w in scales:
        # absolute stick per command at this guidance scale
        null_xy = {}; cmd_xy = {d: {} for d in AX}
        for fp, _ in fr_pool:
            null_xy[fp] = sample_stick_cfg(m, fp, pool_plans["left"][0], 0.0, 0)  # w=0 ignores plan
            for d in AX:
                vals = [sample_stick_cfg(m, fp, p, w, 0) for p in pool_plans[d]]
                cmd_xy[d][fp] = np.mean(vals, axis=0)
        nx = np.mean([null_xy[fp][0] for fp, _ in fr_pool]); ny = np.mean([null_xy[fp][1] for fp, _ in fr_pool])
        print(f"--- w={w} ---  null abs-stick x{nx:+.3f} y{ny:+.3f}")
        steer_ok = 0
        for d in ["left", "right", "up", "down"]:
            mx = np.mean([cmd_xy[d][fp][0] for fp, _ in fr_pool]); my = np.mean([cmd_xy[d][fp][1] for fp, _ in fr_pool])
            ax, want = AX[d]; comp = mx if ax == JLX else my
            ok = np.sign(comp) == want; steer_ok += ok
            tag = "x" if ax == JLX else "y"
            print(f"    {d:6s} x{mx:+.3f} y{my:+.3f}  (abs {tag}{'<0' if want<0 else '>0'} -> {'OK' if ok else 'MISS'})")
        # counterfactual flip: command OPPOSITE of the null's committed axis-dir; absolute flip?
        flip = flip_tot = 0
        for fp, _ in fr_pool:
            nx_, ny_ = null_xy[fp]
            if max(abs(nx_), abs(ny_)) < 0.05:
                continue
            if abs(nx_) >= abs(ny_):
                g = "right" if nx_ > 0 else "left"
            else:
                g = "down" if ny_ > 0 else "up"
            opp = OPP[g]; oax, owant = AX[opp]
            comp = cmd_xy[opp][fp][0] if oax == JLX else cmd_xy[opp][fp][1]
            flip_tot += 1; flip += (np.sign(comp) == owant)
        print(f"    => steering {steer_ok}/4 (absolute);  counterfactual FLIP {flip}/{flip_tot} = {flip/max(flip_tot,1):.0%}\n")


def main(ckpt, which="model"):
    m = load(ckpt, which)
    scales_env = os.environ.get("CFG")
    if scales_env:
        cfg_sweep(m, [float(x) for x in scales_env.split(",")])
        return
    clusters, frames = build()
    rng = np.random.RandomState(0)
    n_plan, n_seed = 5, 2
    pool_plans = {d: [clusters[d][i] for i in rng.choice(len(clusters[d]), min(n_plan, len(clusters[d])), replace=False)]
                  for d in AX}
    # evaluate on a fixed frame pool (cap for runtime)
    fr_pool = frames[:40]
    print(f"frames: {len(fr_pool)}   commands x plans x seeds = 4x{n_plan}x{n_seed}\n")

    # absolute stick per frame under null and each command
    null_xy = {}
    cmd_xy = {d: {} for d in AX}     # cmd_xy[d][fp] = (x,y)
    for fp, _ in fr_pool:
        null_xy[fp] = np.mean([sample_stick(m, fp, "", True, s) for s in range(n_seed)], axis=0)
        for d in AX:
            vals = [sample_stick(m, fp, p, False, s) for p in pool_plans[d] for s in range(n_seed)]
            cmd_xy[d][fp] = np.mean(vals, axis=0)

    # (1) STEERING MATRIX: mean absolute stick (x,y) per command
    print("=== (1) STEERING MATRIX: mean ABSOLUTE left-stick over the frame pool ===")
    print(f"  {'command':9s}  stick-x    stick-y")
    nx = np.mean([null_xy[fp][0] for fp, _ in fr_pool]); ny = np.mean([null_xy[fp][1] for fp, _ in fr_pool])
    print(f"  {'null':9s}  {nx:+.4f}   {ny:+.4f}   (frame-alone / streamer-like)")
    for d in ["left", "right", "up", "down"]:
        mx = np.mean([cmd_xy[d][fp][0] for fp, _ in fr_pool]); my = np.mean([cmd_xy[d][fp][1] for fp, _ in fr_pool])
        ax, want = AX[d]; comp = mx if ax == JLX else my
        tag = "x" if ax == JLX else "y"
        print(f"  {d:9s}  {mx:+.4f}   {my:+.4f}   want {tag}{'<0' if want<0 else '>0'} -> {'OK' if np.sign(comp)==want else 'MISS'}")

    # (2) PLAN-WINS RATE: fraction of (frame,command) where the absolute action goes the
    #     commanded way; and how often it does so even AGAINST the null's tendency on that axis.
    wins = total = 0; override = override_tot = 0
    for d in AX:
        ax, want = AX[d]
        for fp, _ in fr_pool:
            comp = cmd_xy[d][fp][0] if ax == JLX else cmd_xy[d][fp][1]
            ncomp = null_xy[fp][0] if ax == JLX else null_xy[fp][1]
            total += 1; wins += (np.sign(comp) == want)
            # override: null committed to the OPPOSITE side on this axis, command still wins
            if np.sign(ncomp) == -want and abs(ncomp) > 0.05:
                override_tot += 1; override += (np.sign(comp) == want)
    print(f"\n=== (2) PLAN-WINS (absolute action follows the command) ===")
    print(f"  overall plan-wins rate : {wins}/{total} = {wins/total:.0%}")
    print(f"  OVERRIDE rate (null committed OPPOSITE, command still wins): "
          f"{override}/{override_tot} = {override/max(override_tot,1):.0%}")

    # (3) COUNTERFACTUAL FLIP: per frame, command the OPPOSITE of the null's dominant axis-dir
    #     and check the absolute action flips across zero toward the command.
    flip = flip_tot = 0
    for fp, _ in fr_pool:
        nx_, ny_ = null_xy[fp]
        # dominant null axis
        if abs(nx_) >= abs(ny_):
            g = "right" if nx_ > 0 else "left"; ax = JLX
        else:
            g = "down" if ny_ > 0 else "up"; ax = JLY
        if max(abs(nx_), abs(ny_)) < 0.05:
            continue
        opp = OPP[g]; oax, owant = AX[opp]
        comp = cmd_xy[opp][fp][0] if oax == JLX else cmd_xy[opp][fp][1]
        flip_tot += 1; flip += (np.sign(comp) == owant)
    print(f"\n=== (3) COUNTERFACTUAL FLIP (command OPPOSITE of null's committed direction) ===")
    print(f"  absolute-sign flip rate: {flip}/{flip_tot} = {flip/max(flip_tot,1):.0%}")
    print("  (high => the plan reverses a committed frame action => env-free OOD planning.)")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_student/plan_stage1_2500.pt"
    which = sys.argv[2] if len(sys.argv) > 2 else "model"
    main(ckpt, which)
