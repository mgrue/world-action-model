"""Pure 64×64 RGB renderer with supersampled anti-aliasing and a cached background (SPEC §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .physics import CAR_LENGTH, CAR_WIDTH
from .state import State
from .track import WORLD_M, Track, TrackParams

RGB = tuple[int, int, int]

NOSE_FRACTION = 0.35


@dataclass(frozen=True)
class RenderParams:
    grass: RGB = (46, 110, 52)
    road: RGB = (70, 70, 76)
    kerb: RGB = (200, 200, 200)
    body: RGB = (220, 40, 40)
    nose: RGB = (255, 235, 60)
    supersample: int = 4


def _downsample(sub: np.ndarray, ss: int) -> np.ndarray:
    h, w, _ = sub.shape
    avg = sub.reshape(h // ss, ss, w // ss, ss, 3).mean(axis=(1, 3), dtype=np.float64)
    return np.round(avg).astype(np.uint8)


_GRASS, _KERB, _ROAD = 0, 1, 2  # background class labels, also row order of the palette


@lru_cache(maxsize=16)
def _layout(track_params: TrackParams, resolution: int, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    """Colour-independent background: (sub-pixel class labels (n, n) int8,
    per-output-pixel class counts (resolution, resolution, 3) float64). Read-only arrays."""
    track = Track(track_params)
    n = resolution * supersample
    c = (np.arange(n, dtype=np.float64) + 0.5) * (WORLD_M / n)
    px, py = np.meshgrid(c, c)  # px varies along columns (x), py along rows (y)
    d = track.distance_grid(px, py)
    hw = track_params.road_half_width_m
    labels = np.full((n, n), _GRASS, dtype=np.int8)
    labels[d < hw + track_params.kerb_width_m] = _KERB
    labels[d < hw] = _ROAD
    blocks = labels.reshape(resolution, supersample, resolution, supersample)
    counts = np.stack([(blocks == k).sum(axis=(1, 3)) for k in (_GRASS, _KERB, _ROAD)], axis=-1).astype(np.float64)
    labels.flags.writeable = False
    counts.flags.writeable = False
    return labels, counts


def render(state: State, track: Track, params: RenderParams = RenderParams(), resolution: int = 64) -> np.ndarray:
    """Render one Frame: uint8 array of shape (resolution, resolution, 3)."""
    ss = params.supersample
    n = resolution * ss
    labels, counts = _layout(track.params, resolution, ss)
    palette = np.array([params.grass, params.kerb, params.road], dtype=np.float64)
    # Block average of integer colours: the sums are exact in float64, so this equals the
    # per-sub-pixel average bit for bit, for any colours.
    out = np.round(counts @ palette / (ss * ss)).astype(np.uint8)

    # bounding box of the car in sub-pixels, expanded to whole output pixels
    cell = WORLD_M / n
    reach = 0.5 * math.hypot(CAR_LENGTH, CAR_WIDTH)
    j0 = max(0, int(math.floor((state.x - reach) / cell)) // ss * ss)
    j1 = min(n, (int(math.ceil((state.x + reach) / cell)) // ss + 1) * ss)
    i0 = max(0, int(math.floor((state.y - reach) / cell)) // ss * ss)
    i1 = min(n, (int(math.ceil((state.y + reach) / cell)) // ss + 1) * ss)
    if j0 >= j1 or i0 >= i1:
        return out

    xs = (np.arange(j0, j1, dtype=np.float64) + 0.5) * cell - state.x
    ys = (np.arange(i0, i1, dtype=np.float64) + 0.5) * cell - state.y
    fx, fy = math.cos(state.theta), math.sin(state.theta)
    fwd = xs[None, :] * fx + ys[:, None] * fy
    lat = xs[None, :] * -fy + ys[:, None] * fx
    car = (np.abs(fwd) < CAR_LENGTH / 2) & (np.abs(lat) < CAR_WIDTH / 2)
    if not car.any():
        return out
    nose = car & (fwd > CAR_LENGTH / 2 - NOSE_FRACTION * CAR_LENGTH)

    patch = palette[labels[i0:i1, j0:j1]]
    patch[car] = params.body
    patch[nose] = params.nose
    out[i0 // ss : i1 // ss, j0 // ss : j1 // ss] = _downsample(patch, ss)
    return out
