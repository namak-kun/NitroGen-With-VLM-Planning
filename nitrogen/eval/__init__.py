"""Game-agnostic closed-loop evaluation harness for plan-conditioned NitroGen.

Public API:
    from nitrogen.eval import (
        Scenario, Observation, GameEnv, Policy, SuccessDetector,
        EpisodeRunner, EpisodeResult,
        StatePredicateDetector, VLMJudgeDetector,
        EvalProtocol, SuiteResult,
        NullPolicy, ScriptedDirectionPolicy,
    )
    from nitrogen.eval.envs.dummy import DummyGridEnv

See nitrogen/eval/core.py for the design rationale (the "same start, different goals"
counterfactual selection protocol).
"""
from .core import (
    ACTION_DIM, JLX, JLY, JRX, JRY, N_BUTTONS,
    EpisodeResult, EpisodeRunner, GameEnv, Observation, Policy, Scenario, SuccessDetector,
)
from .detectors import StatePredicateDetector, VLMJudgeDetector, SteerDirectionDetector
from .policies import NullPolicy, ScriptedDirectionPolicy, neutral_chunk
from .runner import EvalProtocol, SuiteResult

__all__ = [
    "ACTION_DIM", "JLX", "JLY", "JRX", "JRY", "N_BUTTONS",
    "Scenario", "Observation", "GameEnv", "Policy", "SuccessDetector",
    "EpisodeRunner", "EpisodeResult",
    "StatePredicateDetector", "VLMJudgeDetector", "SteerDirectionDetector",
    "EvalProtocol", "SuiteResult",
    "NullPolicy", "ScriptedDirectionPolicy", "neutral_chunk",
]
