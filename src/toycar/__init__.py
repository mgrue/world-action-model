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
    "A_LAT_MAX", "ACCEL", "BRAKE", "CAR_LENGTH", "CAR_WIDTH", "DELTA_MAX", "DT", "R_MIN", "V_MAX",
    "WHEELBASE", "Branch", "CarEnv", "Direction", "Info", "Projection", "RenderParams", "Route",
    "StartSpec", "State", "StepResult", "Termination", "Track", "TrackParams", "__version__",
    "render", "step_state", "wrap",
]
