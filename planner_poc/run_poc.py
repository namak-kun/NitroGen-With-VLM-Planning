"""run_poc.py — the closed-loop plan-conditioning POC.

THE QUESTION a POC must answer: does the env-free-trained System-2 plan CAUSALLY steer the frozen
System-1 NitroGen agent toward a COMMANDED goal, IN a real game? We answer it with the canonical
"same start, different goals" counterfactual (nitrogen/eval/core.py): from the IDENTICAL start, run
the policy under each directional plan and score it against EVERY direction-goal. If plan_i reaches
goal_i (its own commanded direction) more than goal_j!=i, the plan selects behavior -> capability is
real, measured in a game, not a proxy.

Goal definition here is FRAME-BASED (SteerDirectionDetector): most envs expose no privileged state,
so "did the agent go LEFT/RIGHT/UP/DOWN" is read from net scene optical flow. This needs no game RAM
and generalizes across every env. (Envs that DO expose read_state can swap in StatePredicateDetector.)

Usage:
  env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. .venv/bin/python planner_poc/run_poc.py \
      --env stk --policy nitrogen --ckpt runs/stage2_2b_clean/plan_stage1_2000.pt \
      --cfg 8 --nsteps 10 --axes lr
  # fast harness/detector validation without a GPU:
  ... --policy scripted
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")

from nitrogen.eval import Scenario, EpisodeRunner
from nitrogen.eval.detectors import SteerDirectionDetector
from nitrogen.eval.policies import ScriptedDirectionPolicy

AXES = {
    "lr": ["left", "right"],
    "ud": ["up", "down"],
    "lrud": ["left", "right", "up", "down"],
}
PLAN_TEXT = {
    "left": "go left", "right": "go right", "up": "go up", "down": "go down",
}


# Registry for the remaining built env classes that take the standard (boot_wait, freeze_during_inference)
# constructor. Keeps make_env_factory's explicit if-chain for envs needing special args (stk, custom
# levels, emulators) while making EVERY built env reachable from the recorder/POC harness.
ENV_REGISTRY = {
    "baba": ("baba", "BabaEnv", 14.0),
    "cavestory": ("cavestory", "CaveStoryEnv", 11.0),
    "dustracing": ("dustracing", "DustRacingEnv", 14.0),
    "flare_arpg": ("flare_arpg", "FlareARPGEnv", 14.0),
    "freedink": ("freedink", "FreeDinkEnv", 14.0),
    "ikemen": ("ikemen", "IkemenEnv", 16.0),
    "mrrescue": ("mrrescue", "MrRescueEnv", 12.0),
    "naev": ("naev", "NaevEnv", 16.0),
    "openmw": ("openmw", "OpenMWEnv", 20.0),
    "opentyrian": ("opentyrian", "OpenTyrianEnv", 12.0),
    "parsec47": ("parsec47", "Parsec47Env", 12.0),
    "pekka_kana_2": ("pekka_kana_2", "PekkaKana2Env", 12.0),
    "shattered_pd": ("shattered_pd", "ShatteredPDEnv", 16.0),
    "sokoban": ("sokoban", "SokobanEnv", 12.0),
    "starfighter": ("starfighter", "StarfighterEnv", 12.0),
    "theseeker": ("theseeker", "TheSeekerEnv", 16.0),
    "wesnoth": ("wesnoth", "WesnothEnv", 16.0),
    "zelda_classic": ("zelda_classic", "ZeldaClassicEnv", 14.0),
    "daemon_vs_demon": ("daemon_vs_demon", "DaemonVsDemonEnv", 14.0),
    "mighty_retro_zero": ("mighty_retro_zero", "MightyRetroZeroEnv", 14.0),
    "mmx8": ("mmx8", "MMX8Env", 16.0),
    "megaman_maverick": ("megaman_maverick", "MegamanMaverickEnv", 16.0),
    "floatmancer": ("floatmancer", "FloatmancerEnv", 12.0),
    "platformer_challenge": ("platformer_challenge", "PlatformerChallengeEnv", 12.0),
    "space_rescue_squad": ("space_rescue_squad", "SpaceRescueSquadEnv", 12.0),
}

# Special-cased env names handled explicitly in factory() (custom args / custom levels / emulators).
SPECIAL_ENVS = {"stk", "sdlpop", "thextech", "thextech_get_flower", "notebook_adventure",
                "blind_jump", "castlevania_godot", "xmoto", "trigger_rally", "solarus_zelda",
                "chromium_bsu", "blobwars", "witchblast"}
KNOWN_ENVS = SPECIAL_ENVS | set(ENV_REGISTRY)

# Envs whose build ARTIFACTS were lost (verified by planner_poc/env_healthcheck.py 2026-06-24): they
# raise FileNotFoundError/RuntimeError at construction. Hidden from list_envs() by default so the
# recorder dropdown doesn't surface dead options. Re-fetch/rebuild the artifact to re-enable. The
# value is the missing piece (for whoever restores it).
BROKEN_ENVS = {
    "daemon_vs_demon": "Godot 2.1.6 binary missing (.nitrogen-env-build/godot...)",
    "megaman_maverick": "built LWJGL3 jar missing (.nitrogen-env-build/megaman-maverick/...)",
    "openmw": "OpenMW free Example Suite data missing",
    "theseeker": "The Seeker binary missing (.nitrogen-env-build/TheSeeker...)",
    "zelda_classic": "ZQuest Classic zplayer runtime missing (.nitrogen-env-build/zelda_classic/runtime)",
}


def list_envs(include_broken=False):
    """Env names make_env_factory can build. Excludes envs with lost build artifacts by default
    (see BROKEN_ENVS) so the recorder only offers ones that actually boot."""
    names = KNOWN_ENVS if include_broken else (KNOWN_ENVS - set(BROKEN_ENVS))
    return sorted(names)


def make_env_factory(name, **kw):
    """Return a 0-arg factory that boots a FRESH env (clean same-start per plan)."""
    def factory():
        if name == "stk":
            from nitrogen.eval.envs.supertuxkart import SuperTuxKartEnv
            return SuperTuxKartEnv(track=kw.get("track", "hacienda"), ai=1, laps=1,
                                   boot_wait=kw.get("boot_wait", 20.0), freeze_during_inference=kw.get("freeze", True))
        if name == "sdlpop":
            from nitrogen.eval.envs.sdlpop import SDLPoPEnv
            return SDLPoPEnv(boot_wait=kw.get("boot_wait", 12.0), freeze_during_inference=kw.get("freeze", True))
        if name == "thextech":
            from nitrogen.eval.envs.thextech import TheXTechEnv
            return TheXTechEnv(boot_wait=kw.get("boot_wait", 15.0), freeze_during_inference=kw.get("freeze", True))
        if name == "thextech_get_flower":
            from nitrogen.eval.envs.thextech import TheXTechGetFlowerEnv
            return TheXTechGetFlowerEnv(boot_wait=kw.get("boot_wait", 5.0), freeze_during_inference=kw.get("freeze", True))
        if name == "notebook_adventure":
            from nitrogen.eval.envs.notebook_adventure import NotebookAdventureEnv
            return NotebookAdventureEnv()
        if name == "blind_jump":
            from nitrogen.eval.envs.blind_jump import BlindJumpEnv
            return BlindJumpEnv()
        if name == "castlevania_godot":
            from nitrogen.eval.envs.castlevania_godot import CastlevaniaGodotEnv
            return CastlevaniaGodotEnv(boot_wait=kw.get("boot_wait", 15.0), freeze_during_inference=kw.get("freeze", True))
        if name == "xmoto":
            from nitrogen.eval.envs.xmoto import XMotoEnv
            return XMotoEnv(boot_wait=kw.get("boot_wait", 12.0), freeze_during_inference=kw.get("freeze", True))
        if name == "trigger_rally":
            from nitrogen.eval.envs.trigger_rally import TriggerRallyEnv
            return TriggerRallyEnv(boot_wait=kw.get("boot_wait", 14.0), freeze_during_inference=kw.get("freeze", True))
        if name == "solarus_zelda":
            from nitrogen.eval.envs.solarus_zelda import SolarusZeldaEnv
            return SolarusZeldaEnv(boot_wait=kw.get("boot_wait", 14.0), freeze_during_inference=kw.get("freeze", True))
        if name == "chromium_bsu":
            from nitrogen.eval.envs.chromium_bsu import ChromiumBSUEnv
            return ChromiumBSUEnv(boot_wait=kw.get("boot_wait", 12.0), freeze_during_inference=kw.get("freeze", True))
        if name == "blobwars":
            from nitrogen.eval.envs.blobwars import BlobwarsEnv
            return BlobwarsEnv(boot_wait=kw.get("boot_wait", 12.0), freeze_during_inference=kw.get("freeze", True))
        if name == "witchblast":
            from nitrogen.eval.envs.witchblast import WitchBlastEnv
            return WitchBlastEnv(boot_wait=kw.get("boot_wait", 12.0), freeze_during_inference=kw.get("freeze", True))
        if name in ENV_REGISTRY:
            mod, cls, bw = ENV_REGISTRY[name]
            import importlib
            EnvCls = getattr(importlib.import_module(f"nitrogen.eval.envs.{mod}"), cls)
            return EnvCls(boot_wait=kw.get("boot_wait", bw), freeze_during_inference=kw.get("freeze", True))
        raise ValueError(f"unknown env {name!r}. Known: {', '.join(sorted(KNOWN_ENVS))}")
    return factory


def build_policy(kind, ckpt, cfg, qwen):
    if kind == "scripted":
        return ScriptedDirectionPolicy(magnitude=0.5)
    if kind == "nitrogen":
        from eval_policy import NitroGenPolicy
        pol = NitroGenPolicy(ckpt, qwen=qwen, default_cfg=cfg)
        pol.mm_mode = True; pol.mm_text_only = True
        return pol
    raise ValueError(kind)


def run_counterfactual(env_factory, policy, directions, nsteps, cfg, save_dir):
    """For each commanded direction (a fresh env each), run the policy and accumulate the steering
    progress of EVERY direction-detector. Returns progress matrix P[plan_i][goal_j] and frames."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(directions)
    P = np.zeros((n, n), dtype=np.float32)
    strips = {}
    for i, d in enumerate(directions):
        plan = PLAN_TEXT[d]
        sc = Scenario(id=f"plan_{d}", plan=plan, objective=f"steer {d}", group="cf",
                      max_steps=nsteps, cfg_scale=cfg)
        print(f"\n=== plan='{plan}' (commanded {d}) ===", flush=True)
        env = env_factory()
        # one detector per candidate goal direction, all reading the SAME episode
        dets = []
        for dj in directions:
            det = SteerDirectionDetector()
            det.reset(Scenario(id=f"goal_{dj}", plan="", objective=f"steer {dj}",
                               success_spec={"direction": dj, "flow_thresh": 0.6}))
            dets.append(det)
        try:
            obs = env.reset(sc)
            policy.reset(sc)
            for det in dets:
                det.update(obs)
            frames = [obs.frame]
            for t in range(nsteps):
                action = policy.act(obs)
                obs = env.step(action)
                for det in dets:
                    det.update(obs)
                frames.append(obs.frame)
            progresses = [det.progress() for det in dets]
            P[i] = progresses
            for dj, pr in zip(directions, progresses):
                mark = " <== own" if dj == d else ""
                print(f"    progress[{dj:>5}] = {pr:+.3f}{mark}", flush=True)
            strips[d] = frames
            # save a trajectory strip
            try:
                import cv2
                strip = np.concatenate([cv2.resize(f, (160, 120)) for f in frames[::max(1, len(frames)//8)]], axis=1)
                cv2.imwrite(os.path.join(save_dir, f"poc_{d}.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"    (strip save failed: {e})")
        finally:
            env.close()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
    return P, directions


def summarize(P, directions, save_dir):
    n = len(directions)
    # binary selection: for plan i, did its OWN direction get the max progress?
    argmax_correct = [int(np.argmax(P[i]) == i) for i in range(n)]
    diag = float(np.mean([P[i][i] for i in range(n)]))
    off = float(np.mean([P[i][j] for i in range(n) for j in range(n) if i != j])) if n > 1 else 0.0
    # contrast per plan: own-direction progress minus best competing direction
    contrasts = []
    for i in range(n):
        others = [P[i][j] for j in range(n) if j != i]
        contrasts.append(float(P[i][i] - (max(others) if others else 0.0)))
    summary = {
        "directions": directions,
        "progress_matrix": P.tolist(),
        "argmax_selectivity": float(np.mean(argmax_correct)),  # frac plans whose own dir won
        "mean_diag_progress": diag,
        "mean_offdiag_progress": off,
        "diag_minus_off": diag - off,
        "per_plan_contrast": contrasts,
        "mean_contrast": float(np.mean(contrasts)),
    }
    print("\n===== PLAN-SELECTION SUMMARY =====")
    print("progress matrix P[plan_i][goal_j] (rows=plan, cols=goal-direction):")
    header = "        " + "".join(f"{d:>9}" for d in directions)
    print(header)
    for i, d in enumerate(directions):
        row = "".join(f"{P[i][j]:+9.3f}" for j in range(n))
        print(f"  {d:>5} {row}")
    print(f"\n  argmax_selectivity (own dir wins) = {summary['argmax_selectivity']:.2f}  (1.0 = perfect)")
    print(f"  mean diagonal progress            = {diag:+.3f}")
    print(f"  mean off-diagonal progress        = {off:+.3f}")
    print(f"  diag - off                        = {diag - off:+.3f}  (>0 => plan steers toward its goal)")
    print(f"  mean per-plan contrast            = {summary['mean_contrast']:+.3f}")
    with open(os.path.join(save_dir, "poc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved {save_dir}/poc_summary.json + poc_<dir>.png strips")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="stk")
    ap.add_argument("--policy", default="nitrogen", choices=["nitrogen", "scripted"])
    ap.add_argument("--ckpt", default="runs/stage2_2b_clean/plan_stage1_2000.pt")
    ap.add_argument("--qwen", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--nsteps", type=int, default=10)
    ap.add_argument("--axes", default="lr", choices=list(AXES))
    ap.add_argument("--boot_wait", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    directions = AXES[args.axes]
    save_dir = args.out or f"docs/poc/{args.env}_{args.policy}_{args.axes}"
    kw = {}
    if args.boot_wait is not None:
        kw["boot_wait"] = args.boot_wait
    env_factory = make_env_factory(args.env, **kw)
    print(f"POC: env={args.env} policy={args.policy} axes={directions} cfg={args.cfg} nsteps={args.nsteps}")
    if args.policy == "nitrogen":
        print(f"     ckpt={args.ckpt} qwen={args.qwen}")
    policy = build_policy(args.policy, args.ckpt, args.cfg, args.qwen)
    t0 = time.time()
    P, dirs = run_counterfactual(env_factory, policy, directions, args.nsteps, args.cfg, save_dir)
    summarize(P, dirs, save_dir)
    print(f"\n[done in {time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
