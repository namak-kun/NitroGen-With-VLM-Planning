"""rl_planner_prompt.py -- the RL-time planner OBSERVATION format (@namak-kun's spec, 2026-06-26).

This is the closed-loop prompt the System-2 planner sees AT RL TIME (every A=2 chunks). It differs
from BOTH generate_plan() (flat frame window, no actions) and generate_grounded_plan.py (coarse
per-CHUNK frames+actions): here the observation is interleaved at SUB-CHUNK granularity so the planner
sees, step by step, the inputs the actor took and the frame that resulted -- i.e. cause and effect at
the actor's own intra-chunk sample rate.

User's exact layout (the "input format, revisited"):

    [SYSTEM]  intro + task            (role of the planner)
              game info               (which game / objective)
              controls                (the grounded control vocabulary)
    [USER]
              frame f0                                       <- state before this 2-chunk window
              s1_1   (subchunk 1 of chunk 1, intra-rate=3 -> 6 of the 18 steps)
              f1_1   (frame AFTER subchunk s1_1)
              s1_2
              f1_2
              s1_3
              f1_3   (frame after the ENTIRE chunk 1 elapsed)
              s2_1
              f2_1
              s2_2
              f2_2
              s2_3
              f2_3   (LAST frame; chunk 2 done -> the planner is invoked here. A=2)
              [optional] your last two plans (exploration prior -- ABLATABLE)
              final OUTPUT INSTRUCTIONS

    => the model THINKS (Qwen3.5 think-mode) then emits K=8 plan tokens' worth of plan text.

Key knobs (baked to the RL config the user fixed):
  K = 8      plan tokens per chunk          (PlanHead num_plan_tokens; not used to build the prompt,
                                             recorded here so the trainer stays consistent)
  A = 2      planner invoked every 2 chunks (this prompt spans exactly A chunks -> ends at f{A}_{S})
  S = 3      intra-chunk sample rate        (subchunks per chunk -> subchunk_len = 18 // 3 = 6)

THINK MODE (crucial, user): we generate with the chat template's enable_thinking=True so the model's
analysis lands inside <think>...</think> and the PLAN OUTPUT stays clean. generate_rl_plan() splits on
'</think>' and returns ONLY the post-think plan (plus the think trace, for logging/RL-credit). Without
this, "model analysis and all will fall into the actual output and make the plan diluted."

DEFERRAL (user's System-1<->System-2 idea, scaffolded -- NEEDS RL signal we don't yet have): with
allow_defer=True the output instructions permit the model to emit the sentinel DEFER_PHRASE
("NO GUIDANCE NEEDED") when the situation is purely reactive and a high-level plan won't help. The
caller maps that to a null/masked plan (base-DiT-exact). This is the "noncommittal plan / special
token" deferral the user preferred over a model-controlled CFG weight. Off by default (ablatable).

PREV PLANS (user: "put in the last two plans... to incentivize exploration but idk -- maybe ablate"):
include_prev_plans=True appends the last <=2 plans so the planner can avoid repeating a stuck plan.
Off by default (ablatable).

Run (synthetic demo -- no env, proves think-mode separation end to end):
  env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B \
    .venv/bin/python planner_poc/rl_planner_prompt.py --demo --genre platformer
  # just print the assembled prompt (no VLM):
  ... --demo --print-prompt-only
  # build from real NitroGen chunk dirs (each actions_*.parquet) + their frames:
  ... --chunk-dir <c1> <c2> --frames-dir <frames> --genre platformer
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.training.actions import load_chunk_actions, summarize_chunk

# Reuse the SAME grounded-control vocabulary + output instructions as the teacher prompt so the two
# stay consistent (no duplicated constants drifting apart).
from generate_grounded_plan import CONTROLS, OUTPUT_INSTRUCTIONS

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")

DEFER_PHRASE = "NO GUIDANCE NEEDED"


@dataclass
class RLPlannerConfig:
    """The RL planner interface constants the user fixed (K=8, A=2, intra-rate=3)."""
    num_plan_tokens: int = 8        # K -- plan tokens emitted per chunk (trainer/PlanHead side)
    num_chunks: int = 2             # A -- planner invoked every A chunks; this prompt spans A chunks
    intra_chunk_rate: int = 3       # S -- subchunks per chunk (frames sampled within a chunk)
    action_horizon: int = 18        # steps per chunk (NitroGen)

    @property
    def subchunk_len(self) -> int:
        return max(1, self.action_horizon // self.intra_chunk_rate)


# ---- System header tailored to the SUB-CHUNK interleaved RL observation. ---------------------------
SYS_INTRO = (
    "You are the high-level planner (System 2) for an agent playing a game. A fast low-level "
    "controller (System 1) turns your plan into button/stick inputs. You are invoked periodically; "
    "each time you see the recent play so far as an interleaved trace: the game frame, then the "
    "inputs the controller actually took over a short sub-interval, then the frame that resulted, and "
    "so on. Use this cause-and-effect trace (you CANNOT read the controller from a frame alone) to "
    "judge whether the agent is making progress or is stuck/looping, and decide what to do next."
)


def _subchunk_dict(chunk: dict, lo: int, hi: int) -> dict:
    """Slice [lo:hi] out of a loaded chunk-action dict so summarize_chunk can describe a subchunk."""
    out = {}
    for k in ("buttons", "j_left", "j_right"):
        if k in chunk and chunk[k] is not None:
            out[k] = np.asarray(chunk[k])[lo:hi]
    return out


def subchunk_summaries(chunk: dict, n_sub: int, subchunk_len: int) -> list[str]:
    """Per-subchunk action summaries for one 18-step chunk: n_sub strings, each describing
    `subchunk_len` steps' worth of stick motion + buttons (reusing summarize_chunk on the slice)."""
    H = int(np.asarray(chunk["buttons"]).shape[0])
    out = []
    for s in range(n_sub):
        lo = s * subchunk_len
        hi = H if s == n_sub - 1 else min(H, (s + 1) * subchunk_len)
        sub = _subchunk_dict(chunk, lo, hi)
        # summarize_chunk wants its own n_seg; for a 6-step subchunk 1 segment is plenty.
        out.append(summarize_chunk(sub, n_seg=1))
    return out


def build_rl_messages(
    f0: Image.Image | np.ndarray | None,
    chunks: list[dict],
    *,
    genre: str = "generic",
    game_info: str = "",
    cfg: RLPlannerConfig | None = None,
    prev_plans: list[str] | None = None,
    include_prev_plans: bool = False,
    allow_defer: bool = False,
):
    """Assemble the interleaved RL planner observation.

    f0       : the frame BEFORE this A-chunk window (state f0). May be None (text-only fallback).
    chunks   : list of A chunk records, time-ordered. Each record is a dict:
                 {"subs": [(action, frame), (action, frame), (action, frame)]}    # len == S
               where `action` is EITHER a pre-made summary str OR a loaded chunk-action dict slice,
               and `frame` is the PIL/np frame AFTER that subchunk (the last sub's frame = chunk end).
               Convenience: a record may instead be {"chunk": <full chunk-action dict>,
               "frames": [f_sub1, f_sub2, f_sub3]} and we summarize the subchunks ourselves.
    Returns (system_text, content, images, render_text) in the Qwen chat-content format.
    """
    cfg = cfg or RLPlannerConfig()
    S = cfg.intra_chunk_rate
    controls = CONTROLS.get(genre, CONTROLS["generic"])
    sys_parts = [SYS_INTRO]
    if game_info:
        sys_parts.append(game_info.strip())
    sys_parts.append(controls)
    system_text = "\n".join(sys_parts)

    content: list[dict] = []
    images: list[Image.Image] = []
    render: list[str] = [f"[SYSTEM]\n{system_text}", "", "[USER]"]

    def _img(x):
        return x if isinstance(x, Image.Image) else Image.fromarray(np.asarray(x)).convert("RGB")

    def add_text(t: str):
        content.append({"type": "text", "text": t})
        render.append(t)

    def add_image(im, tag: str):
        content.append({"type": "image"})
        images.append(_img(im))
        render.append(f"<image: {tag}>")

    # f0 -- state before the window.
    add_text("Current situation (frame f0, before this interval's inputs):")
    if f0 is not None:
        add_image(f0, "f0")

    # Per chunk: interleave (subchunk action text, resulting frame).
    for ci, rec in enumerate(chunks, 1):
        if "subs" in rec:
            subs = rec["subs"]
            sub_actions = [a for (a, _f) in subs]
            sub_frames = [f for (_a, f) in subs]
            sub_actions = [a if isinstance(a, str) else summarize_chunk(a, n_seg=1)
                           for a in sub_actions]
        else:  # {"chunk": full dict, "frames": [...]}
            sub_actions = subchunk_summaries(rec["chunk"], S, cfg.subchunk_len)
            sub_frames = rec.get("frames", [None] * S)
        for si in range(len(sub_actions)):
            add_text(f"Chunk {ci}, inputs s{ci}_{si+1}: {sub_actions[si]}")
            fr = sub_frames[si] if si < len(sub_frames) else None
            tag = f"f{ci}_{si+1}" + (" (chunk end)" if si == len(sub_actions) - 1 else "")
            add_text(f"Resulting frame f{ci}_{si+1}"
                     + (" (after the full chunk elapsed):" if si == len(sub_actions) - 1 else ":"))
            if fr is not None:
                add_image(fr, tag)

    # Optional exploration prior: the last <=2 plans (ablatable).
    if include_prev_plans and prev_plans:
        last2 = [p for p in prev_plans if p][-2:]
        if last2:
            add_text("Your recent plans (avoid repeating one that did not make progress): "
                     + " | ".join(f"({i+1}) {p}" for i, p in enumerate(last2)))

    # Output instructions (+ optional deferral sentinel).
    out_instr = OUTPUT_INSTRUCTIONS
    if allow_defer:
        out_instr = out_instr.rstrip(".") + (
            f". If the situation is purely reactive and a high-level plan would NOT help the "
            f"controller (it can handle this moment on its own), output EXACTLY '{DEFER_PHRASE}' "
            f"instead of a plan."
        )
    add_text(out_instr)

    return system_text, content, images, "\n".join(render)


def parse_think(gen: str) -> tuple[str, str]:
    """Split a think-mode generation into (think_trace, plan). The chat template injects the opening
    '<think>' into the prompt, so the raw generation is '<reasoning...></think>\\n\\nPLAN'. If no
    '</think>' is present (think disabled / model skipped), the whole thing is the plan."""
    if "</think>" in gen:
        think, _, after = gen.partition("</think>")
        think = think.split("<think>")[-1].strip()
        plan = after.strip()
    else:
        think, plan = "", gen.strip()
    plan = plan.strip().strip('"').strip()
    # keep the plan to a single line (the broad plan sentence)
    plan = plan.split("\n")[0].strip() if plan else plan
    return think, plan


def generate_rl_plan(
    planner, system_text: str, content: list[dict], images: list[Image.Image], device: str,
    *, enable_thinking: bool = True, think_budget: int = 256, plan_budget: int = 48,
    max_new_tokens: int | None = None, temperature: float = 0.0, top_p: float = 0.9,
) -> dict:
    """Run the frozen planner VLM on the RL observation in THINK MODE (budget-forced) and return
    {"think": ..., "plan": ..., "raw": ..., "forced": bool}. `planner` is a loaded
    nitrogen.planner.PlanEncoder (has .backbone/.processor). temperature>0 samples (RL exploration).

    BUDGET FORCING (why): Qwen3.5-2B reasons usefully but is verbose and often does NOT close
    </think> within a small budget, so a naive single generate leaks raw analysis into the plan
    (the user's exact worry). We instead: (1) generate up to `think_budget` tokens of reasoning;
    (2) if the model already closed </think> and emitted a plan, use it; (3) otherwise FORCE-close by
    appending '</think>\\n\\n' and generate up to `plan_budget` tokens for the clean plan. This bounds
    planner compute per call (important when the planner is invoked every A=2 chunks in RL) and
    guarantees the analysis stays inside <think>. `max_new_tokens` (if given) overrides think_budget.
    enable_thinking=False = ablation (template emits an empty think block; single short generate)."""
    import torch
    if planner.processor is None:
        raise RuntimeError("generate_rl_plan requires a VL processor (Qwen3VLProcessor + torchvision).")
    if next(planner.backbone.parameters()).device != torch.device(device):
        planner.backbone.to(device)
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": content}]
    try:
        text = planner.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except TypeError:  # processor that doesn't accept the kwarg
        text = planner.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    proc_kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        proc_kwargs["images"] = images
    inp = planner.processor(**proc_kwargs).to(device)
    inlen = inp["input_ids"].shape[1]
    img_kw = {k: inp[k] for k in ("pixel_values", "image_grid_thw") if k in inp}

    def _sample_kwargs():
        if temperature and temperature > 0:
            return dict(do_sample=True, temperature=float(temperature), top_p=float(top_p))
        return dict(do_sample=False)

    if not enable_thinking:
        with torch.no_grad():
            out = planner.backbone.generate(**inp, max_new_tokens=plan_budget or 48, **_sample_kwargs())
        raw = planner.processor.batch_decode(out[:, inlen:], skip_special_tokens=True)[0]
        think, plan = parse_think(raw)
        return {"think": think, "plan": plan, "raw": raw, "forced": False}

    # ---- phase 1: think ----
    tb = max_new_tokens or think_budget
    with torch.no_grad():
        out1 = planner.backbone.generate(**inp, max_new_tokens=tb, **_sample_kwargs())
    gen1 = planner.processor.batch_decode(out1[:, inlen:], skip_special_tokens=False)[0]
    if "</think>" in gen1:
        think, _, after = gen1.partition("</think>")
        think = think.split("<think>")[-1].strip()
        plan = after
        if plan.strip():                       # think closed AND plan present -> done
            _, plan = parse_think("</think>" + plan)
            return {"think": think, "plan": plan, "raw": gen1, "forced": False}
    else:
        think = gen1.split("<think>")[-1].strip()

    # ---- phase 2: force-close </think> and decode the clean plan ----
    # Seed a NEUTRAL "Plan:" lead-in (no direction words) so the model emits a fresh plan sentence
    # rather than completing the (possibly mid-sentence) reasoning we just truncated.
    tok = getattr(planner.processor, "tokenizer", planner.processor)
    close_ids = tok("</think>\n\nPlan:", add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    full_ids = torch.cat([out1, close_ids], dim=1)
    attn = torch.ones_like(full_ids)
    p2 = dict(input_ids=full_ids, attention_mask=attn, max_new_tokens=plan_budget, **_sample_kwargs())
    p2.update(img_kw)
    if "mm_token_type_ids" in inp:             # extend image-type mask with text(0) for the new tokens
        pad = torch.zeros((1, full_ids.shape[1] - inp["mm_token_type_ids"].shape[1]),
                          dtype=inp["mm_token_type_ids"].dtype, device=device)
        p2["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], pad], dim=1)
    with torch.no_grad():
        out2 = planner.backbone.generate(**p2)
    plan = planner.processor.batch_decode(out2[:, full_ids.shape[1]:], skip_special_tokens=True)[0]
    plan = plan.strip().strip('"').strip()
    if plan.lower().startswith("plan:"):
        plan = plan[5:].strip()
    plan = plan.split("\n")[0].strip()
    return {"think": think, "plan": plan, "raw": gen1 + "</think>\n\nPlan:" + plan, "forced": True}


# ----------------------------------------------------------------------------------------------------
def _synthetic_chunks(cfg: RLPlannerConfig, genre: str):
    """A tiny synthetic A-chunk history (gray frames + stick/button summaries) so the demo runs with
    no env/data -- enough to verify prompt assembly + think-mode separation end to end."""
    rng = np.random.default_rng(0)
    f0 = Image.fromarray(rng.integers(40, 60, (180, 256, 3), dtype=np.uint8))
    summaries = {
        "platformer": [["left stick: right; buttons: no buttons",
                        "left stick: right; buttons: JUMP briefly",
                        "left stick: right; buttons: no buttons"],
                       ["left stick: right; buttons: no buttons",
                        "left stick: neutral; buttons: JUMP briefly",
                        "left stick: right; buttons: no buttons"]],
    }.get(genre, [["left stick: right; buttons: no buttons"] * cfg.intra_chunk_rate] * cfg.num_chunks)
    chunks = []
    for ci in range(cfg.num_chunks):
        subs = []
        for si in range(cfg.intra_chunk_rate):
            fr = Image.fromarray(rng.integers(40, 80, (180, 256, 3), dtype=np.uint8))
            subs.append((summaries[ci][si], fr))
        chunks.append({"subs": subs})
    return f0, chunks


def _real_chunks(chunk_dirs, frames_dir, cfg: RLPlannerConfig):
    """Build A chunk records from real NitroGen chunk dirs. Frames are best-effort (uses f*.png in the
    chunk dir if present, else None -> text-only grounding)."""
    f0 = None
    chunks = []
    for cdir in chunk_dirs[: cfg.num_chunks]:
        pq = os.path.join(cdir, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(cdir, "actions_raw.parquet")
        chunk = load_chunk_actions(pq) if os.path.exists(pq) else None
        local = sorted(glob.glob(os.path.join(cdir, "f*.png")))
        frames = []
        if local:
            sel = np.linspace(0, len(local) - 1, num=cfg.intra_chunk_rate).round().astype(int)
            frames = [Image.open(local[i]).convert("RGB") for i in sel]
            if f0 is None:
                f0 = Image.open(local[0]).convert("RGB")
        if chunk is not None:
            chunks.append({"chunk": chunk, "frames": frames or [None] * cfg.intra_chunk_rate})
    return f0, chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use a synthetic A-chunk history (no env/data)")
    ap.add_argument("--chunk-dir", nargs="+", default=None, help="real NitroGen chunk dirs (>=A of them)")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--genre", default="platformer", choices=sorted(CONTROLS))
    ap.add_argument("--game-info", default="Game: a 2D platformer. Objective: advance RIGHT and reach the level exit.")
    ap.add_argument("--include-prev-plans", action="store_true")
    ap.add_argument("--allow-defer", action="store_true")
    ap.add_argument("--no-think", action="store_true", help="ablation: disable think mode")
    ap.add_argument("--print-prompt-only", action="store_true", help="assemble + print, do NOT load the VLM")
    ap.add_argument("--think-budget", type=int, default=256)
    ap.add_argument("--plan-budget", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    cfg = RLPlannerConfig()
    if args.demo or not args.chunk_dir:
        f0, chunks = _synthetic_chunks(cfg, args.genre)
    else:
        f0, chunks = _real_chunks(args.chunk_dir, args.frames_dir, cfg)

    system_text, content, images, render = build_rl_messages(
        f0, chunks, genre=args.genre, game_info=args.game_info, cfg=cfg,
        prev_plans=["move right and jump the gap", "go right along the platform"],
        include_prev_plans=args.include_prev_plans, allow_defer=args.allow_defer)

    n_img = len(images)
    print(f"# RL planner observation: A={cfg.num_chunks} chunks, S={cfg.intra_chunk_rate} subchunks/chunk, "
          f"K={cfg.num_plan_tokens} plan tokens, {n_img} image(s), genre={args.genre}\n")
    print(render)
    print("\n# (think mode:", ("OFF" if args.no_think else "ON"),
          "| prev_plans:", args.include_prev_plans, "| defer:", args.allow_defer, ")")

    if args.print_prompt_only:
        return

    import torch
    from nitrogen.planner import PlanEncoder, PlannerConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN))
    pl.load()
    res = generate_rl_plan(pl, system_text, content, images, device,
                           enable_thinking=not args.no_think, think_budget=args.think_budget,
                           plan_budget=args.plan_budget, temperature=args.temperature)
    print("\n===== THINK TRACE =====\n" + (res["think"] or "(none)"))
    print("\n===== PLAN (clean output)  [forced-close=%s] =====\n" % res["forced"] + res["plan"])
    if args.allow_defer and res["plan"].strip().upper().startswith(DEFER_PHRASE):
        print("\n[defer] planner chose to DEFER -> map to null/masked plan (base-DiT-exact).")


if __name__ == "__main__":
    main()
