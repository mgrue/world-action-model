"""PROTOTYPE — throwaway. Ticket 03: render with a cached class-label grid (grass/kerb/road) and colours
applied per call. Claim: bit-identical to toycar.render for any RenderParams, and fast for new colours.

  uv run python .scratch/step3-dataset/prototypes/recolour_render.py
"""

import math
import time
from functools import lru_cache

import numpy as np

from toycar import CAR_LENGTH, CAR_WIDTH, RenderParams, State, Track, render
from toycar.render import NOSE_FRACTION, _downsample
from toycar.track import WORLD_M

GRASS, KERB, ROAD = 0, 1, 2


@lru_cache(maxsize=8)
def labels(track_params, resolution, ss):
    track = Track(track_params)
    n = resolution * ss
    c = (np.arange(n, dtype=np.float64) + 0.5) * (WORLD_M / n)
    px, py = np.meshgrid(c, c)
    d = track.distance_grid(px, py)
    hw = track_params.road_half_width_m
    lab = np.full((n, n), GRASS, np.int8)
    lab[d < hw + track_params.kerb_width_m] = KERB
    lab[d < hw] = ROAD
    # per output pixel: count of each class among ss×ss sub-pixels
    counts = np.stack([(lab == k).reshape(resolution, ss, resolution, ss).sum(axis=(1, 3)) for k in (GRASS, KERB, ROAD)], -1)
    lab.flags.writeable = counts.flags.writeable = False
    return lab, counts.astype(np.float64)


def render2(state, track, params=RenderParams(), resolution=64):
    ss = params.supersample
    n = resolution * ss
    lab, counts = labels(track.params, resolution, ss)
    pal = np.array([params.grass, params.kerb, params.road], np.float64)
    out = np.round(counts @ pal / (ss * ss)).astype(np.uint8)
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
    patch = pal[lab[i0:i1, j0:j1]]
    patch[car] = params.body
    patch[nose] = params.nose
    out[i0 // ss : i1 // ss, j0 // ss : j1 // ss] = _downsample(patch, ss)
    return out


if __name__ == "__main__":
    tr = Track.default()
    rng = np.random.default_rng(0)
    mism = 0
    for i in range(3000):
        cols = [tuple(int(v) for v in rng.integers(0, 256, 3)) for _ in range(5)]
        p = RenderParams(*cols)
        s = State(float(rng.uniform(0, 40)), float(rng.uniform(0, 40)), float(rng.uniform(-math.pi, math.pi)), 0.0)
        mism += not np.array_equal(render(s, tr, p), render2(s, tr, p))
        render.__wrapped__ if hasattr(render, "__wrapped__") else None
    print("random palettes × poses: mismatches", mism, "of 3000")
    t = time.perf_counter()
    for i in range(500):
        render2(State(20, 27.5, 0, 5), tr, RenderParams(grass=(46, 110, i % 256)))
    print("new-colour frame: %.3f ms" % (1e3 * (time.perf_counter() - t) / 500))
