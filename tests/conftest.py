import math

import pytest

from toycar import DELTA_MAX, WHEELBASE, CarEnv, Track, wrap


@pytest.fixture(scope="session")
def track() -> Track:
    return Track.default()


def pursuit_action(env: CarEnv, route: str, direction: str, speed: float = 5.0, lookahead: float = 4.0):
    """Simple pure-pursuit controller along a Route, for tests only."""
    s = env.state
    r = env.track.route(route)
    st = r.project(s.x, s.y).station_m
    tx, ty, _ = r.pose_at(st + (lookahead if direction == "cw" else -lookahead))
    alpha = wrap(math.atan2(ty - s.y, tx - s.x) - s.theta)
    kappa = 2.0 * math.sin(alpha) / lookahead
    steer = max(-1.0, min(1.0, math.atan(kappa * WHEELBASE) / DELTA_MAX))
    throttle = max(-1.0, min(1.0, (speed - s.v) * 2.0))
    return steer, throttle


TINY_EPISODES = {"train": 12, "val": 6, "test": 6}
TINY_LENGTH = 60


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory):
    """A tiny recorded Step 3 dataset (Step 4 tests)."""
    from toycar import dataset as D

    return D.record(tmp_path_factory.mktemp("tiny") / "tiny", episodes=TINY_EPISODES, length=TINY_LENGTH, workers=2)


TINY_VAE = {"channels": [8, 8, 16, 16], "blocks": 1, "z": 16}


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    """A Checkpoint root holding a tiny, untrained `vae-v1` (Steps 5–8 tests)."""
    import json

    torch = pytest.importorskip("torch")
    from wam import compressor as C

    root = tmp_path_factory.mktemp("ck")
    (root / "vae-v1").mkdir()
    torch.manual_seed(0)
    vae = C.VAE(TINY_VAE)
    torch.save({"state_dict": vae.state_dict(), "arch": vae.arch, "latent_mean": torch.zeros(16),
                "latent_std": torch.ones(16)}, root / "vae-v1" / "model.pt")
    (root / "vae-v1" / "card.json").write_text(json.dumps({"name": "vae-v1", "code_sha256": C.code_sha256()}))
    return root
