"""Value types shared by physics, track, render and env (SPEC §4, §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

Branch = Literal["outer", "inner"]
Direction = Literal["cw", "ccw"]
Termination = Literal["crash", "max_steps"] | None


@dataclass(frozen=True)
class State:
    """Canonical car State: position (m), heading (rad, (−π, π]), speed (m/s)."""

    x: float
    y: float
    theta: float
    v: float

    def features(self) -> np.ndarray:
        """Model-facing State features (x/40, y/40, cos θ, sin θ, v/10), float32 (5,)."""
        return np.array(
            [self.x / 40.0, self.y / 40.0, math.cos(self.theta), math.sin(self.theta), self.v / 10.0],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class StartSpec:
    """Start-pose overrides for CarEnv.reset; None means sample it."""

    direction: Direction | None = None
    route: Branch | None = None
    station_m: float | None = None
    lateral_m: float | None = None
    heading_offset: float | None = None  # rad, relative to the Route tangent in the episode Direction
    speed: float | None = None


@dataclass(frozen=True)
class Info:
    station_m: float
    progress_m: float
    laps: int
    direction: Direction
    wrong_way: bool
    branch: Branch | None
    last_branch: Branch | None
    steps: int
    termination: Termination


@dataclass(frozen=True)
class StepResult:
    state: State
    terminated: bool
    truncated: bool
    info: Info
