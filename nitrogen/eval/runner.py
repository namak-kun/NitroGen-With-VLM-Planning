"""Run scenario suites and aggregate the metrics that matter for plan-conditioning.

The headline metric is PLAN-SELECTION (a.k.a. counterfactual contrast): for a group of
scenarios that share an initial state but carry different plans+objectives, does each plan
reach ITS OWN objective more often than it reaches the OTHER objectives? If plan A reaches
goal A but not goal B (and vice versa) from the identical start, the plan causally selects
among valid futures — the planning capability is real, measured in a game, not a proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .core import EpisodeResult, EpisodeRunner, GameEnv, Policy, Scenario, SuccessDetector


@dataclass
class SuiteResult:
    results: list[EpisodeResult]

    @property
    def success_rate(self) -> float:
        return sum(r.success for r in self.results) / max(len(self.results), 1)

    def by_group(self) -> dict[str, list[EpisodeResult]]:
        g: dict[str, list[EpisodeResult]] = {}
        for r in self.results:
            grp = r.extra.get("group", "")
            g.setdefault(grp, []).append(r)
        return g


class EvalProtocol:
    """Runs (scenario x policy) episodes and computes plan-selection metrics. `detector_factory`
    returns a FRESH SuccessDetector per episode (detectors are stateful)."""

    def __init__(self, env: GameEnv, detector_factory, runner: Optional[EpisodeRunner] = None):
        self.env = env
        self.detector_factory = detector_factory
        self.runner = runner or EpisodeRunner()

    def run_suite(self, policy: Policy, scenarios: list[Scenario],
                  repeats: int = 1, verbose: bool = True) -> SuiteResult:
        results: list[EpisodeResult] = []
        for sc in scenarios:
            for rep in range(repeats):
                det = self.detector_factory()
                res = self.runner.run(self.env, policy, det, sc)
                res.extra["group"] = sc.group
                res.extra["objective"] = sc.objective
                res.extra["rep"] = rep
                results.append(res)
                if verbose:
                    print(f"  [{sc.id} rep{rep}] success={res.success} "
                          f"steps={res.steps} ({res.reason})", flush=True)
        return SuiteResult(results)

    @staticmethod
    def plan_selection_matrix(env: GameEnv, policy: Policy, detector_factory,
                              group_scenarios: list[Scenario], runner: Optional[EpisodeRunner] = None):
        """The counterfactual core: for scenarios sharing one start (a `group`), run the policy
        under EACH plan, and evaluate against EVERY scenario's success criterion. Returns an
        NxN matrix M[plan_i][objective_j] = success. The diagonal (own plan -> own objective)
        should beat the off-diagonal if the plan selects behavior. Identical start is guaranteed
        because all scenarios in the group share `init`.
        """
        runner = runner or EpisodeRunner()
        n = len(group_scenarios)
        M = [[False] * n for _ in range(n)]
        for i, sc_plan in enumerate(group_scenarios):
            # run ONE episode under plan i, but check it against all objectives j in parallel
            dets = [detector_factory() for _ in range(n)]
            obs = env.reset(sc_plan)
            policy.reset(sc_plan)
            for j in range(n):
                dets[j].reset(group_scenarios[j])
                dets[j].update(obs)
            for t in range(sc_plan.max_steps):
                if all(d.status()[0] for d in dets):
                    break
                action = policy.act(obs)
                obs = env.step(action)
                for j in range(n):
                    if not dets[j].status()[0]:
                        dets[j].update(obs)
                if obs.done:
                    break
            for j in range(n):
                M[i][j] = dets[j].status()[1]
        return M

    @staticmethod
    def selection_score(M) -> dict:
        """Summarize a plan-selection matrix: diagonal hit-rate (own plan reaches own goal) vs
        off-diagonal (own plan reaches a DIFFERENT goal). High diagonal + low off-diagonal =
        the plan selects behavior."""
        n = len(M)
        diag = sum(M[i][i] for i in range(n))
        off = sum(M[i][j] for i in range(n) for j in range(n) if i != j)
        off_total = n * (n - 1)
        return {
            "n": n,
            "diag_success": diag, "diag_rate": diag / max(n, 1),
            "off_success": off, "off_rate": off / max(off_total, 1),
            "selectivity": (diag / max(n, 1)) - (off / max(off_total, 1)),
        }
