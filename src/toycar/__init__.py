"""toycar: deterministic 2D overhead car simulator with a 64×64 RGB renderer."""

from .env import CarEnv
from .physics import (
    A_LAT_MAX,
    ACCEL,
    BRAKE,
    CAR_LENGTH,
    CAR_WIDTH,
    DELTA_MAX,
    DT,
    R_MIN,
    V_MAX,
    WHEELBASE,
    step_state,
    wrap,
)
from .render import RenderParams, render
from .state import Branch, Direction, Info, StartSpec, State, StepResult, Termination
from .track import Projection, Route, Track, TrackParams

__version__ = "0.1.0"

__all__ = [
    "A_LAT_MAX", "ACCEL", "INSTRUCTIONS", "EpisodeScore", "Expert", "ExpertAction", "ExpertParams",
    "Instruction", "score_episode", "summarize", "BRAKE", "CAR_LENGTH", "CAR_WIDTH", "DELTA_MAX", "DT", "R_MIN", "V_MAX",
    "WHEELBASE", "Branch", "CarEnv", "Direction", "Info", "Projection", "RenderParams", "Route",
    "StartSpec", "State", "StepResult", "Termination", "Track", "TrackParams", "__version__",
    "render", "step_state", "wrap",
]


_EXPERT_NAMES = frozenset(
    ["INSTRUCTIONS", "EpisodeScore", "Expert", "ExpertAction", "ExpertParams", "Instruction", "score_episode", "summarize"]
)


def __getattr__(name: str):
    # Lazy re-export, so `python -m toycar.expert` doesn't import the module twice.
    if name in _EXPERT_NAMES:
        from . import expert

        return getattr(expert, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
