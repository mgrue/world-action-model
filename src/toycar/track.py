"""Track geometry: two closed Routes made of exact line and arc primitives (SPEC §2–3)."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from .physics import CAR_LENGTH, CAR_WIDTH, R_MIN, wrap
from .state import Branch, State

WORLD_M = 40.0
_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TrackParams:
    center: tuple[float, float] = (20.0, 20.0)
    straight_m: float = 18.0  # length of top and bottom straights
    end_radius_m: float = 7.5  # radius of left and right end arcs
    fork_length_m: float = 18.0  # chord length of the fork (must equal straight_m)
    bulge_m: float = 6.0  # apex offset of each branch from the straight's line
    road_half_width_m: float = 2.5
    kerb_width_m: float = 0.3


@dataclass(frozen=True)
class Projection:
    station_m: float  # s of the closest centerline point on this Route
    lateral_m: float  # signed offset along the Route's right-hand normal (+ = right of the cw tangent)
    distance_m: float  # |lateral_m|


@dataclass(frozen=True)
class _Prim:
    """One primitive. Lines: sign == 0. Arcs: sign = ±1 (+ = clockwise), radius R, centre (cx, cy)."""

    s0: float
    length: float
    x0: float
    y0: float
    th0: float
    sign: int
    radius: float
    cx: float
    cy: float

    def pose(self, u: float) -> tuple[float, float, float]:
        if self.sign == 0:
            return (self.x0 + u * math.cos(self.th0), self.y0 + u * math.sin(self.th0), self.th0)
        th = self.th0 + self.sign * u / self.radius
        sr = self.sign * self.radius
        return (self.cx + sr * math.sin(th), self.cy - sr * math.cos(th), th)

    def curvature(self) -> float:
        return 0.0 if self.sign == 0 else self.sign / self.radius

    def candidates(self, x: float, y: float) -> list[float]:
        """Local arc lengths u of candidate closest points, in increasing u."""
        if self.sign == 0:
            u = (x - self.x0) * math.cos(self.th0) + (y - self.y0) * math.sin(self.th0)
            return [min(max(u, 0.0), self.length)]
        out = [0.0]
        qx, qy = x - self.cx, y - self.cy
        if qx != 0.0 or qy != 0.0:
            th = math.atan2(qy, qx) + self.sign * math.pi / 2
            u = self.radius * ((self.sign * (th - self.th0)) % _TWO_PI)
            if u <= self.length:
                out.append(u)
        out.append(self.length)
        return out

    def distance_grid(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Vectorised distance from points to this primitive (used for background rendering)."""
        if self.sign == 0:
            c, s = math.cos(self.th0), math.sin(self.th0)
            u = np.clip((px - self.x0) * c + (py - self.y0) * s, 0.0, self.length)
            return np.hypot(px - (self.x0 + u * c), py - (self.y0 + u * s))
        qx, qy = px - self.cx, py - self.cy
        r = np.hypot(qx, qy)
        th = np.arctan2(qy, qx) + self.sign * math.pi / 2
        u = self.radius * np.mod(self.sign * (th - self.th0), _TWO_PI)
        x1, y1, _ = self.pose(self.length)
        d_end = np.minimum(np.hypot(px - self.x0, py - self.y0), np.hypot(px - x1, py - y1))
        return np.where(u <= self.length, np.abs(r - self.radius), d_end)


class Route:
    """One closed centerline. Station s is arc length from the Fork, clockwise."""

    def __init__(self, name: Branch, start: tuple[float, float, float], spec: list[tuple]) -> None:
        self.name: Branch = name
        prims: list[_Prim] = []
        x, y, th = start
        s = 0.0
        for item in spec:
            if item[0] == "line":
                length = item[1]
                p = _Prim(s, length, x, y, th, 0, 0.0, 0.0, 0.0)
            else:
                _, radius, turn = item
                sign = 1 if turn > 0 else -1
                length = radius * abs(turn)
                # centre lies along the right normal r(θ) = (−sin θ, cos θ) for a cw turn
                cx = x + sign * radius * -math.sin(th)
                cy = y + sign * radius * math.cos(th)
                p = _Prim(s, length, x, y, th, sign, radius, cx, cy)
            prims.append(p)
            x, y, th = p.pose(p.length)
            s += length
        if not (math.hypot(x - start[0], y - start[1]) < 1e-9 and abs(wrap(th - start[2])) < 1e-9):
            raise AssertionError(f"route {name} does not close: end=({x}, {y}, {th})")
        self._prims = prims
        self._starts = [p.s0 for p in prims]
        self.length_m: float = s

    def _prim_at(self, s: float) -> tuple[_Prim, float]:
        s = s % self.length_m
        i = bisect.bisect_right(self._starts, s) - 1
        p = self._prims[i]
        return p, min(s - p.s0, p.length)

    def pose_at(self, s: float) -> tuple[float, float, float]:
        """(x, y, cw tangent heading in (−π, π]); s taken mod length."""
        p, u = self._prim_at(s)
        x, y, th = p.pose(u)
        return x, y, wrap(th)

    def curvature_at(self, s: float) -> float:
        p, _ = self._prim_at(s)
        return p.curvature()

    def project(self, x: float, y: float) -> Projection:
        best_d = math.inf
        best: tuple[float, float, float, float] | None = None  # (s, px, py, th)
        for p in self._prims:
            for u in p.candidates(x, y):
                px, py, th = p.pose(u)
                d = math.hypot(x - px, y - py)
                if d < best_d:  # strict: ties go to the lower Station
                    best_d = d
                    best = (p.s0 + u, px, py, th)
        assert best is not None
        s, px, py, th = best
        if s >= self.length_m:
            s -= self.length_m
        side = (x - px) * -math.sin(th) + (y - py) * math.cos(th)
        return Projection(s, math.copysign(best_d, side) if best_d > 0 else 0.0, best_d)

    def distance_grid(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        d = np.full(np.broadcast(px, py).shape, np.inf)
        for p in self._prims:
            np.minimum(d, p.distance_grid(px, py), out=d)
        return d


class Track:
    def __init__(self, params: TrackParams = TrackParams()) -> None:
        self.params = params
        p = params
        if p.fork_length_m != p.straight_m:
            raise ValueError("fork_length_m must equal straight_m")
        alpha = 2.0 * math.atan(2.0 * p.bulge_m / p.fork_length_m)
        rho = p.fork_length_m / (4.0 * math.sin(alpha))
        if not rho > R_MIN:
            raise ValueError(f"branch arc radius {rho:.3f} m must exceed R_MIN {R_MIN:.3f} m")
        reach = p.road_half_width_m + p.kerb_width_m
        if not 2.0 * p.bulge_m - 2.0 * reach > 0:
            raise ValueError("island is closed: bulge too small for the road width")

        cx, cy = p.center
        start = (cx - p.straight_m / 2.0, cy - p.end_radius_m, 0.0)
        tail = [("arc", p.end_radius_m, math.pi), ("line", p.straight_m), ("arc", p.end_radius_m, math.pi)]
        self.outer = Route("outer", start, [("arc", rho, -alpha), ("arc", rho, 2 * alpha), ("arc", rho, -alpha), *tail])
        self.inner = Route("inner", start, [("arc", rho, alpha), ("arc", rho, -2 * alpha), ("arc", rho, alpha), *tail])
        self.length_m: float = self.outer.length_m
        self.fork_station_m: float = 0.0
        self.merge_station_m: float = 4.0 * rho * alpha
        if abs(self.inner.length_m - self.outer.length_m) > 1e-9:
            raise AssertionError("routes have different lengths")

        # inner branch apex vs bottom straight: kerbs must not touch
        inner_apex_y = start[1] + p.bulge_m
        bottom_y = cy + p.end_radius_m
        if not (bottom_y - inner_apex_y) - 2.0 * reach > 0:
            raise ValueError("inner branch kerb overlaps the bottom straight's kerb")

        # whole road incl. kerb inside the frame (analytic extents)
        extents = [
            cx - p.straight_m / 2.0 - p.end_radius_m,  # left end arc
            cx + p.straight_m / 2.0 + p.end_radius_m,  # right end arc
            start[1] - p.bulge_m,  # outer apex
            bottom_y,  # bottom straight
        ]
        x_lo, x_hi, y_lo, y_hi = extents
        if x_lo - reach < 0 or y_lo - reach < 0 or x_hi + reach > WORLD_M or y_hi + reach > WORLD_M:
            raise ValueError("road and kerb must lie within the 40 m × 40 m world")

    @classmethod
    def default(cls) -> Track:
        return cls(TrackParams())

    def route(self, name: Branch) -> Route:
        if name == "outer":
            return self.outer
        if name == "inner":
            return self.inner
        raise ValueError(f"unknown route {name!r}")

    def locate(self, x: float, y: float) -> tuple[float, Branch | None]:
        po = self.outer.project(x, y)
        pi = self.inner.project(x, y)
        name: Branch
        proj, name = (po, "outer") if po.distance_m <= pi.distance_m else (pi, "inner")
        s = proj.station_m
        in_fork = self.fork_station_m <= s < self.merge_station_m
        return s, (name if in_fork else None)

    def distance_to_centerline(self, x: float, y: float) -> float:
        return min(self.outer.project(x, y).distance_m, self.inner.project(x, y).distance_m)

    def distance_grid(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Vectorised distance_to_centerline over arrays of points."""
        return np.minimum(self.outer.distance_grid(px, py), self.inner.distance_grid(px, py))

    def car_corners(self, state: State) -> list[tuple[float, float]]:
        fx, fy = math.cos(state.theta), math.sin(state.theta)
        rx, ry = -fy, fx
        hl, hw = CAR_LENGTH / 2.0, CAR_WIDTH / 2.0
        return [
            (state.x + a * hl * fx + b * hw * rx, state.y + a * hl * fy + b * hw * ry)
            for a in (1.0, -1.0)
            for b in (1.0, -1.0)
        ]

    def is_crash(self, state: State) -> bool:
        limit = self.params.road_half_width_m + self.params.kerb_width_m
        return any(self.distance_to_centerline(cx, cy) > limit for cx, cy in self.car_corners(state))
