"""Regenerate tests/golden/physics_trajectory.npz (SPEC §9). Run once and commit:

    uv run python tests/golden/generate.py
"""

import math
from pathlib import Path

import numpy as np

from toycar import State, step_state

START = State(20.0, 27.5, math.pi, 5.0)


def trajectory(actions: np.ndarray) -> np.ndarray:
    s = START
    rows = [[s.x, s.y, s.theta, s.v]]
    for a in actions:
        s = step_state(s, a)
        rows.append([s.x, s.y, s.theta, s.v])
    return np.array(rows, dtype=np.float64)


if __name__ == "__main__":
    actions = np.random.default_rng(0).uniform(-1, 1, (300, 2))
    out = Path(__file__).with_name("physics_trajectory.npz")
    np.savez(out, actions=actions, states=trajectory(actions))
    print(f"wrote {out}")
