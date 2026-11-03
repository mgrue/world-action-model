"""Stateful episode wrapper: start poses, progress/lap bookkeeping, termination (SPEC §6)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import numpy as np

from .physics import step_state, wrap
from .render import RenderParams, render
from .state import Branch, Direction, Info, StartSpec, State, StepResult, Termination
from .track import Track

WRONG_WAY_M = 5.0
MAX_RESET_ATTEMPTS = 1000


class CarEnv:
    def __init__(
        self,
        track: Track | None = None,
        render_params: RenderParams | None = None,
        max_steps: int = 900,
        resolution: int = 64,
    ) -> None:
        self._track = track if track is not None else Track.default()
        self._default_params = render_params if render_params is not None else RenderParams()
        self._params = self._default_params
        self.max_steps = max_steps
        self.resolution = resolution
        self._state: State | None = None
        self._ended = True

    # -- properties -------------------------------------------------------------------------
    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("call reset() first")
        return self._state

    @property
    def track(self) -> Track:
        return self._track

    @property
    def render_params(self) -> RenderParams:
        return self._params

    # -- episode ----------------------------------------------------------------------------
    def _sample_start(self, rng: np.random.Generator, start: StartSpec) -> tuple[State, Direction]:
        track = self._track
        direction = cast(Direction, ["cw", "ccw"][int(rng.integers(2))])
        route = cast(Branch, ["outer", "inner"][int(rng.integers(2))])
        station = float(rng.uniform(0.0, track.length_m))
        lateral = float(rng.uniform(-1.0, 1.0))
        heading_offset = float(rng.uniform(-math.pi / 18, math.pi / 18))
        speed = float(rng.uniform(0.0, 6.0))

        if start.direction is not None:
            direction = start.direction
        if start.route is not None:
            route = start.route
        if start.station_m is not None:
            station = float(start.station_m)
        if start.lateral_m is not None:
            lateral = float(start.lateral_m)
        if start.heading_offset is not None:
            heading_offset = float(start.heading_offset)
        if start.speed is not None:
            speed = float(start.speed)

        px, py, t = track.route(route).pose_at(station)
        x = px + lateral * -math.sin(t)
        y = py + lateral * math.cos(t)
        theta = wrap(t + (math.pi if direction == "ccw" else 0.0) + heading_offset)
        return State(x, y, theta, speed), direction

    def reset(
        self, seed: int, start: StartSpec | None = None, render_params: RenderParams | None = None
    ) -> tuple[State, Info]:
        rng = np.random.default_rng(seed)
        spec = start if start is not None else StartSpec()
        for _ in range(MAX_RESET_ATTEMPTS):
            state, direction = self._sample_start(rng, spec)
            if not self._track.is_crash(state):
                break
        else:
            raise ValueError(f"no valid start pose after {MAX_RESET_ATTEMPTS} attempts for {spec}")

        self._state = state
        self._direction: Direction = direction
        self._params = render_params if render_params is not None else self._default_params
        self._steps = 0
        self._progress = 0.0
        self._max_progress = 0.0
        self._laps = 0
        self._wrong_way = False
        self._station, self._branch = self._track.locate(state.x, state.y)
        self._last_branch: Branch | None = None
        # which end the car entered the fork section from: +1 = Fork (moving cw), −1 = Merge, 0 = unknown
        self._branch_entry = 0
        self._termination: Termination = None
        self._ended = False
        return state, self._info()

    def _info(self) -> Info:
        return Info(
            station_m=self._station,
            progress_m=self._progress,
            laps=self._laps,
            direction=self._direction,
            wrong_way=self._wrong_way,
            branch=self._branch,
            last_branch=self._last_branch,
            steps=self._steps,
            termination=self._termination,
        )

    def step(self, action: Sequence[float]) -> StepResult:
        if self._ended:
            raise RuntimeError("episode ended; call reset()")
        track = self._track
        L = track.length_m

        state = step_state(self.state, action)
        self._state = state
        self._steps += 1

        s_new, branch_new = track.locate(state.x, state.y)
        delta = s_new - self._station
        if delta > L / 2:
            delta -= L
        elif delta < -L / 2:
            delta += L
        self._progress += delta if self._direction == "cw" else -delta
        self._max_progress = max(self._max_progress, self._progress)
        self._laps = max(0, math.floor(self._progress / L))
        self._wrong_way = self._progress < self._max_progress - WRONG_WAY_M

        prev = self._branch
        if prev is None and branch_new is not None:
            self._branch_entry = 1 if delta > 0 else -1
        elif prev is not None and branch_new is None:
            exit_sense = 1 if delta > 0 else -1  # +1 = past the Merge, −1 = past the Fork
            # completed the passage (left at the far end), not backed out the way it came
            if self._branch_entry == 0 or exit_sense == self._branch_entry:
                self._last_branch = prev
            self._branch_entry = 0
        self._station, self._branch = s_new, branch_new

        terminated = truncated = False
        if track.is_crash(state):
            terminated = True
            self._termination = "crash"
        elif self._steps >= self.max_steps:
            truncated = True
            self._termination = "max_steps"
        self._ended = terminated or truncated
        return StepResult(state, terminated, truncated, self._info())

    def render(self) -> np.ndarray:
        return render(self.state, self._track, self._params, self.resolution)
