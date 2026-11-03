"""Drive recordings (.npz) and the replay tool (SPEC §7).

    uv run python -m toycar.replay FILE [--show] [--export-png DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .env import CarEnv
from .render import RenderParams
from .state import StartSpec, State
from .track import Track, TrackParams


def frame_sha256(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def state_row(s: State) -> list[float]:
    return [s.x, s.y, s.theta, s.v]


@dataclass
class Recording:
    """One episode being recorded; `states` and `hashes` include the reset State's entry."""

    seed: int
    start: StartSpec
    render_params: RenderParams
    track_params: TrackParams
    actions: list[tuple[float, float]] = dataclasses.field(default_factory=list)
    states: list[list[float]] = dataclasses.field(default_factory=list)
    hashes: list[str] = dataclasses.field(default_factory=list)

    def add(self, state: State, frame: np.ndarray, action: Sequence[float] | None = None) -> None:
        if action is not None:
            self.actions.append((float(action[0]), float(action[1])))
        self.states.append(state_row(state))
        self.hashes.append(frame_sha256(frame))

    def save(self, directory: str | Path, termination: str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": self.seed,
            "start": dataclasses.asdict(self.start),
            "render_params": dataclasses.asdict(self.render_params),
            "track_params": dataclasses.asdict(self.track_params),
            "toycar_version": __version__,
            "termination": termination,
        }
        path = directory / f"drive_{self.seed}.npz"
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(meta)),
            actions=np.array(self.actions, dtype=np.float64).reshape(-1, 2),
            states=np.array(self.states, dtype=np.float64).reshape(-1, 4),
            frame_sha256=np.array(self.hashes),
        )
        return path


def _tuples(d: dict) -> dict:
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def load(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        return meta, z["actions"], z["states"], z["frame_sha256"]


def env_from_meta(meta: dict) -> tuple[CarEnv, StartSpec, RenderParams]:
    track = Track(TrackParams(**_tuples(meta["track_params"])))
    params = RenderParams(**_tuples(meta["render_params"]))
    start = StartSpec(**meta["start"])
    return CarEnv(track=track, render_params=params, max_steps=10**9), start, params


def verify(path: str | Path) -> tuple[bool, str]:
    """Replay a recording headlessly and compare States (exact) and frame hashes."""
    meta, actions, states, hashes = load(path)
    env, start, params = env_from_meta(meta)
    state, _ = env.reset(meta["seed"], start, params)

    def check(k: int, s: State) -> str | None:
        got = state_row(s)
        for name, a, b in zip(("x", "y", "theta", "v"), got, states[k]):
            if a != float(b):
                return f"FAIL at tick {k}: {name} {a!r} != {float(b)!r}"
        if frame_sha256(env.render()) != str(hashes[k]):
            return f"FAIL at tick {k}: frame hash differs"
        return None

    if len(states) != len(actions) + 1 or len(hashes) != len(states):
        return False, "FAIL: inconsistent array lengths"
    if (msg := check(0, state)) is not None:
        return False, msg
    for k, action in enumerate(actions, start=1):
        result = env.step(action)
        if (msg := check(k, result.state)) is not None:
            return False, msg
        if (result.terminated or result.truncated) and k != len(actions):
            return False, f"FAIL at tick {k}: episode ended early ({result.info.termination})"
    return True, "PASS"


def frames(path: str | Path):
    """Yield the replayed 64×64 frames (reset frame first)."""
    meta, actions, _, _ = load(path)
    env, start, params = env_from_meta(meta)
    env.reset(meta["seed"], start, params)
    yield env.render()
    for action in actions:
        env.step(action)
        yield env.render()


def export_png(path: str | Path, out_dir: str | Path) -> int:
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, frame in enumerate(frames(path)):
        pygame.image.save(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), str(out / f"frame_{k:05d}.png"))
        n += 1
    return n


def show(path: str | Path) -> None:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((512, 512))
    pygame.display.set_caption(f"toycar replay — {Path(path).name}")
    clock = pygame.time.Clock()
    try:
        for frame in frames(path):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
                ):
                    return
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (512, 512)), (0, 0))
            pygame.display.flip()
            clock.tick(15)
    finally:
        pygame.quit()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m toycar.replay", description=__doc__.splitlines()[0])
    ap.add_argument("file")
    ap.add_argument("--show", action="store_true", help="play the drive back in a window at 15 Hz")
    ap.add_argument("--export-png", metavar="DIR", help="write 64×64 frames as PNGs")
    args = ap.parse_args(argv)

    ok, msg = verify(args.file)
    print(msg)
    if args.export_png:
        print(f"wrote {export_png(args.file, args.export_png)} frames to {args.export_png}")
    if args.show:
        show(args.file)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
