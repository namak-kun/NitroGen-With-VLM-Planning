"""Smoke test for the eval harness on the DummyGridEnv — NO game/model needed. Validates:
  (1) Scenario -> Env -> Policy -> Detector -> Runner produces a sane EpisodeResult.
  (2) The "same start, different goals" PLAN-SELECTION protocol: from an identical start, a
      scripted direction-policy driven by the plan text reaches the goal in the planned
      direction and NOT the others -> selectivity ~ 1.0. This is the miniature of the real
      counterfactual planning test the FOSS-game eval will run with NitroGen.
Run: python planner_poc/eval_harness_smoke.py
"""
import sys
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))

from nitrogen.eval import (
    EpisodeRunner, EvalProtocol, ScriptedDirectionPolicy, Scenario,
    StatePredicateDetector,
)
from nitrogen.eval.envs.dummy import DummyGridEnv

SIZE = 16
START = {"x": SIZE / 2, "y": SIZE / 2, "goals": {
    "west": (1, 8), "east": (14, 8), "north": (8, 1), "south": (8, 14)}}

# Four scenarios sharing the SAME start; each plan should drive to its own edge goal.
# reach_region radius 2 over the privileged state keys x,y.
def make_scenarios():
    spec = lambda cx, cy: {"reach_region": ("x", "y", cx, cy, 2.0)}
    return [
        Scenario("go_west",  "move left toward the west marker",  "reach west",
                 init=START, success_spec=spec(1, 8),  max_steps=60, group="hub"),
        Scenario("go_east",  "move right toward the east marker", "reach east",
                 init=START, success_spec=spec(14, 8), max_steps=60, group="hub"),
        Scenario("go_north", "move up toward the north marker",   "reach north",
                 init=START, success_spec=spec(8, 1),  max_steps=60, group="hub"),
        Scenario("go_south", "move down toward the south marker", "reach south",
                 init=START, success_spec=spec(8, 14), max_steps=60, group="hub"),
    ]


def main():
    env = DummyGridEnv(size=SIZE)
    policy = ScriptedDirectionPolicy(magnitude=1.0)
    scenarios = make_scenarios()

    print("=== (1) basic suite: each plan vs its own objective ===")
    proto = EvalProtocol(env, StatePredicateDetector)
    suite = proto.run_suite(policy, scenarios)
    print(f"  suite success_rate = {suite.success_rate:.2f} (expect 1.00)\n")

    print("=== (2) PLAN-SELECTION matrix (same start, different goals) ===")
    M = EvalProtocol.plan_selection_matrix(env, policy, StatePredicateDetector, scenarios)
    names = [s.id for s in scenarios]
    print("        " + "  ".join(f"{n[:6]:>6s}" for n in names) + "   (objective ->)")
    for i, row in enumerate(M):
        print(f"  {names[i]:>8s} " + "  ".join(f"{'  OK  ' if v else '  .   '}" for v in row))
    score = EvalProtocol.selection_score(M)
    print(f"\n  diag (own plan -> own goal): {score['diag_success']}/{score['n']} "
          f"= {score['diag_rate']:.2f}")
    print(f"  off  (own plan -> other goal): {score['off_success']}/{score['n']*(score['n']-1)} "
          f"= {score['off_rate']:.2f}")
    print(f"  SELECTIVITY (diag - off): {score['selectivity']:+.2f}  (expect ~ +1.00)")
    ok = score["diag_rate"] > 0.99 and score["off_rate"] < 0.01 and suite.success_rate > 0.99
    print(f"\n  HARNESS {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
