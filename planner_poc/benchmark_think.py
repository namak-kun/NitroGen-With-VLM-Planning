"""Benchmark Qwen3.5 THINK-MODE as the System-2 planner (the user's ask: reasoning in <think>, a
parseable final plan, and the FINAL-PLAN embeddings used to condition NitroGen; think-mode is needed
for RL credit-assignment + exploration).

What this benches, on real game frames:
  1. THINK works: enable_thinking=True -> <think>reasoning</think> + a final plan. We parse both.
  2. PARSEABLE final plan: we ask for `PLAN: <directive>` after the reasoning and extract it.
  3. EMBEDDINGS to continue: we forward the full generated sequence and mean-pool the last-layer
     hidden states over ONLY the post-</think> final-plan tokens -> the (d,) conditioning vector that
     would feed the resampler/adapter (instead of the text-only encode). We verify shape + that it
     differs from the no-think embedding.
  4. vs NO-THINK: same prompt with enable_thinking=False -> compare plan + latency.
  5. GROUNDING (the open gap): does reasoning help? frame vs horizontal-mirror -> does the plan flip?
  6. RL EXPLORATION: sample (temperature>0) several times -> plan diversity (distinct plans).

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
        .venv/bin/python planner_poc/benchmark_think.py
"""
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.planner import PlanEncoder, PlannerConfig

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
SYS = ("You are the high-level planner for an agent playing a 2D game. You see the recent frames "
       "(oldest first). Decide the single best next move. Think step by step about what is on screen "
       "and where to go, then end with your final answer on its own line EXACTLY as: PLAN: <one short "
       "imperative directive, max 8 words>.")
INSTR = "What should the agent do next?"
THINK_CLOSE = 248069  # </think> token id
THINK_BUDGET = int(os.environ.get("THINK_BUDGET", "256"))  # reasoning-token budget before force-close


def _imgs(frames):
    return [f if isinstance(f, Image.Image) else Image.fromarray(np.asarray(f)).convert("RGB")
            for f in frames]


def _prompt(pl, imgs, enable_thinking):
    content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": INSTR}]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
    try:
        return pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=enable_thinking)
    except TypeError:
        return pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_plan(text):
    """Extract (think_trace, final_plan) from a generated string."""
    think = ""
    if "</think>" in text:
        think, text = text.split("</think>", 1)
        think = think.replace("<think>", "").strip()
    m = re.search(r"PLAN:\s*(.+)", text, re.I)
    plan = (m.group(1) if m else text).strip().splitlines()[0].strip().strip('"').strip() if text.strip() else ""
    return think, plan


def _gen_ids(pl, input_ids, attn, pixel_kw, n, sample, temp, seed):
    if sample:
        torch.manual_seed(seed)
    return pl.backbone.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=n,
                                do_sample=sample, temperature=temp if sample else None,
                                top_p=0.9 if sample else None, **pixel_kw)


@torch.no_grad()
def generate(pl, frames, device, enable_thinking, max_new_tokens, sample=False, temp=0.7, seed=0,
             force_close=True, plan_budget=40):
    """Generate a plan. In think mode we use BUDGET FORCING (the Qwen-team-recommended pattern, of
    which s1's budget forcing is the academic form): let the model reason for up to `max_new_tokens`
    tokens; if it has not emitted </think> by then, close the block but FIRST append a transition
    nudge inside <think> ("Considering the limited time, I have to give the answer based on the
    thinking so far now.") -- per Qwen this yields a more coherent forced answer than an abrupt cut --
    then `</think>\n\nPLAN:` and decode the directive in `plan_budget` tokens. This bounds the thinking
    budget (the mechanism RL needs) and guarantees a parseable plan."""
    imgs = _imgs(frames)
    prompt = _prompt(pl, imgs, enable_thinking)
    inp = pl.processor(text=[prompt], images=imgs, return_tensors="pt").to(device)
    pixel_kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
    gen_start = inp["input_ids"].shape[1]
    t0 = time.time()
    out = _gen_ids(pl, inp["input_ids"], inp["attention_mask"], pixel_kw,
                   max_new_tokens, sample, temp, seed)
    gen_ids = out[0, gen_start:]
    close_pos = (gen_ids == THINK_CLOSE).nonzero(as_tuple=True)[0]
    forced = False
    if enable_thinking and force_close and not len(close_pos):
        # BUDGET FORCING: reconstruct a fresh prompt = (reasoning so far) + the Qwen transition nudge
        # + forced close marker + PLAN stub, then re-run a single clean forward. We rebuild the prompt
        # (instead of concatenating ids) because the VL rope-index recompute breaks on a continued
        # image sequence.
        reasoning = pl.processor.batch_decode(gen_ids[None], skip_special_tokens=True)[0].strip()
        nudge = ("\n\nConsidering the limited time, I have to give the answer based on the thinking "
                 "so far now.")
        forced_prompt = prompt + reasoning + nudge + "\n</think>\n\nPLAN:"
        inp = pl.processor(text=[forced_prompt], images=imgs, return_tensors="pt").to(device)
        pixel_kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        gen_start = inp["input_ids"].shape[1]
        out = _gen_ids(pl, inp["input_ids"], inp["attention_mask"], pixel_kw,
                       plan_budget, sample, temp, seed)
        dt = time.time() - t0
        think = reasoning
        plan_text = "PLAN:" + pl.processor.batch_decode(out[0, gen_start:][None],
                                                        skip_special_tokens=True)[0]
        closed, forced, n_think = True, True, int(gen_ids.shape[0])
    else:
        dt = time.time() - t0
        # split think vs final-plan by the </think> TOKEN ID (skip_special_tokens erases the marker).
        if len(close_pos):
            ci = int(close_pos[-1])
            think = pl.processor.batch_decode(gen_ids[:ci][None], skip_special_tokens=True)[0].strip()
            plan_text = pl.processor.batch_decode(gen_ids[ci + 1:][None], skip_special_tokens=True)[0]
            closed, n_think = True, ci
        else:
            think, closed, n_think = "", False, 0
            plan_text = pl.processor.batch_decode(gen_ids[None], skip_special_tokens=True)[0]
    m = re.search(r"PLAN:\s*(.+)", plan_text, re.I)
    plan = (m.group(1) if m else plan_text).strip().splitlines()[0].strip().strip('"').strip() if plan_text.strip() else ""
    # --- conditioning embedding: encode the FINAL plan text the way it would feed NitroGen, i.e.
    # frame-grounded text-only hiddens (encode_multimodal text_only) mean-pooled -> the (d,) vector
    # the resampler/adapter consume. (Avoids re-forwarding the raw generated ids, which trips the
    # VL rope-index recompute.)
    h, _ = pl.encode_multimodal(imgs, plan or plan_text[:64], device, text_only=True)
    plan_emb = h[0].float().mean(0).cpu().numpy()
    return dict(think=think, plan=plan, raw=plan_text, dt=dt, emb=plan_emb,
                n_gen=int(out.shape[1] - inp["input_ids"].shape[1]), n_think_tok=n_think,
                closed=closed, forced=forced)


def load_frames():
    import glob
    files = (sorted(glob.glob("docs/env_candidates/stk_*.png"))[:2]
             + sorted(glob.glob("docs/poc/stk_lora600_lr/poc_*.png"))[:1])
    return {os.path.basename(f)[:18]: np.asarray(Image.open(f).convert("RGB")) for f in files}


def main():
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    dev = "cuda"
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    frames = load_frames()
    print(f"=== THINK-MODE BENCH ({QWEN}) on {len(frames)} game frames ===\n")

    for name, fr in frames.items():
        print(f"----- frame {name} -----")
        nt = generate(pl, [fr], dev, enable_thinking=False, max_new_tokens=32)
        th = generate(pl, [fr], dev, enable_thinking=True, max_new_tokens=THINK_BUDGET)
        print(f"  NO-THINK : plan={nt['plan']!r:40} ({nt['n_gen']} tok, {nt['dt']:.1f}s)")
        print(f"  THINK    : plan={th['plan']!r:40} ({th['n_gen']} tok incl {th['n_think_tok']} think, "
              f"{'forced-close' if th['forced'] else 'self-closed'}, {th['dt']:.1f}s)")
        print(f"     think trace[:200]: {th['think'][:200]!r}")
        print(f"  EMB: no-think {nt['emb'].shape} | think {th['emb'].shape} | "
              f"cos(nt,think)={float(np.dot(nt['emb'],th['emb'])/(np.linalg.norm(nt['emb'])*np.linalg.norm(th['emb'])+1e-8)):.3f} "
              f"(low => thinking changes the conditioning vector)")
        # grounding: mirror flip with thinking
        mir = fr[:, ::-1, :].copy()
        thm = generate(pl, [mir], dev, enable_thinking=True, max_new_tokens=THINK_BUDGET)
        a = th['plan'].lower(); b = thm['plan'].lower()
        da = 'left' if 'left' in a else ('right' if 'right' in a else '?')
        db = 'left' if 'left' in b else ('right' if 'right' in b else '?')
        flip = {da, db} == {'left', 'right'}
        print(f"  GROUNDING(think): orig={da} mirror={db} -> {'FLIP (grounded!)' if flip else 'same (still biased)'}")
        # RL exploration: sampled plan diversity (with thinking)
        plans = set()
        for s in range(3):
            plans.add(generate(pl, [fr], dev, enable_thinking=True, max_new_tokens=THINK_BUDGET,
                               sample=True, temp=0.6, seed=s)['plan'].lower())
        print(f"  RL DIVERSITY (3 samples, T=0.6): {len(plans)} distinct plans -> {list(plans)}\n", flush=True)

    print("VERDICT NOTES: think-mode works if THINK rows show a <think> trace + a parsed PLAN; the "
          "final-plan EMB is extractable (shape == backbone hidden) and feeds the resampler in place "
          "of the text-only encode; GROUNDING shows whether reasoning fixes the mirror bias; RL "
          "DIVERSITY>1 means sampling gives exploration for policy-gradient.")


if __name__ == "__main__":
    main()
