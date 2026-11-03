import json

import numpy as np

from toycar import CarEnv, StartSpec
from toycar import replay
from toycar.replay import Recording


def _record(tmp_path, n=120):
    start = StartSpec(direction="cw", route="outer", speed=0.0)
    env = CarEnv(max_steps=10**9)
    s, _ = env.reset(11, start)
    rec = Recording(11, start, env.render_params, env.track.params)
    rec.add(s, env.render())
    rng = np.random.default_rng(3)
    for _ in range(n):
        a = (float(rng.uniform(-0.3, 0.3)), float(rng.uniform(0.0, 1.0)))
        r = env.step(a)
        rec.add(r.state, env.render(), a)
        if r.terminated:
            break
    return rec.save(tmp_path, "crash" if r.terminated else "quit")


def test_replay_roundtrip(tmp_path):
    path = _record(tmp_path)
    assert path.name == "drive_11.npz"
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        assert z["actions"].dtype == np.float64 and z["actions"].shape[1] == 2
        assert z["states"].shape == (len(z["actions"]) + 1, 4)
        assert z["frame_sha256"].shape == (len(z["states"]),)
    assert set(meta) == {"seed", "start", "render_params", "track_params", "toycar_version", "termination"}
    assert replay.main([str(path)]) == 0


def test_replay_detects_tampering(tmp_path):
    path = _record(tmp_path)
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["actions"] = data["actions"].copy()
    data["actions"][10, 0] += 1e-6
    np.savez_compressed(path, **data)
    ok, msg = replay.verify(path)
    assert not ok and msg.startswith("FAIL at tick 11")


def test_export_png(tmp_path):
    path = _record(tmp_path, n=5)
    n = replay.export_png(path, tmp_path / "png")
    assert n == 6 and (tmp_path / "png" / "frame_00005.png").exists()
