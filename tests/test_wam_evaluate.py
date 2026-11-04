"""Steps 5–8 evaluation and CLI (SPEC §12 tests 19–22)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toycar.dataset import open_dataset  # noqa: E402
from wam import evaluate as E  # noqa: E402
from wam import policy as P  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
H, S = 16, 16


def synthetic(n=60, seed=0):
    """References, entry offsets, Instructions and Planned branches for n synthetic Fork Windows."""
    rng = np.random.default_rng(seed)
    ref = np.full((17, 2, H), np.nan)
    for e in range(1, 17):
        ref[e, 0] = np.linspace(0.52, -0.78, H)
        ref[e, 1] = np.linspace(0.57, 0.82, H)
    entry = rng.integers(1, 13, n)
    instr = np.tile([0, 1, 2], n // 3)
    planned = np.where(instr == 2, rng.integers(0, 2, n), instr)
    return ref, entry, instr, planned, rng


def check(steer, entry, instr, ref):
    f = E.fork_metrics(steer, entry, instr, ref)
    g = E.gates({"rmse_steer": 0.0, "rmse_throttle": 0.0}, f)
    return f, {k: v["pass"] for k, v in g.items()}


def test_metrics_on_fakes():  # 19
    ref, entry, instr, planned, rng = synthetic()
    noise = lambda: rng.normal(0, 0.03, (len(entry), S, H))  # noqa: E731
    # labels: outer/inner follow the Instruction; `any` samples split between both Branches
    branch = np.where(instr[:, None] == 2, rng.integers(0, 2, (len(entry), S)), instr[:, None])
    branch[instr == 2, 0], branch[instr == 2, 1] = 0, 1
    labels = ref[entry[:, None], branch] + noise()
    f, ok = check(labels, entry, instr, ref)
    assert all(ok.values()), f
    assert f["commit"] > 0.99 and f["follow"] == 1.0 and f["any_two_sided"] == 1.0

    avg = np.broadcast_to(0.5 * (ref[entry, 0] + ref[entry, 1])[:, None], labels.shape)
    f, ok = check(avg, entry, instr, ref)
    assert f["commit"] == 0.0 and not ok["commit"]

    shuffled = ref[entry[:, None], np.broadcast_to(rng.integers(0, 2, len(entry))[:, None], (len(entry), S))] + noise()
    f, ok = check(shuffled, entry, instr, ref)
    assert not ok["follow"] and 0.2 < f["follow"] < 0.8

    collapse = ref[entry[:, None], np.where(instr[:, None] == 2, 0, instr[:, None]) + np.zeros((1, S), int)] + noise()
    f, ok = check(collapse, entry, instr, ref)
    assert ok["commit"] and ok["follow"] and not ok["any_two_sided"] and not ok["any_balance"]

    act = np.random.default_rng(1).uniform(-1, 1, (30, H, 2))
    g = E.general_metrics(np.repeat(act[:, None], 4, 1), act)
    assert g["rmse_steer"] == 0 and g["rmse_throttle"] == 0 and g["best_rmse_steer"] == 0
    b = E.baselines(act, (0.0, 0.077))
    assert b["shuffled"]["rmse_steer"] > b["zero"]["rmse_steer"] * 0.5


def test_subsets_deterministic(tiny_data):  # 20
    ds = open_dataset(tiny_data, check=False)
    sz = E.sizes(64)
    a, b = E.subsets(ds, 16, sz), E.subsets(ds, 16, sz)
    for k in ("general", "fork", "periodic_general", "periodic_fork"):
        assert np.array_equal(getattr(a, k).index, getattr(b, k).index), k
    assert a.hash == b.hash
    assert not getattr(a, "general").approaches_section.any()
    assert E.subsets(ds, 8, sz).hash != a.hash and E.subsets(ds, 16, E.sizes(32)).hash != a.hash
    r1, r2 = E.fork_references(ds), E.fork_references(ds)
    assert np.array_equal(r1, r2, equal_nan=True) and r1.shape == (17, 2, 16)
    assert E.fork_references(ds, 8).shape == (17, 2, 8)
    assert E.eval_hash(a, r1) == E.eval_hash(b, r2)


def test_cli_smoke(tiny_data, tiny_root, tmp_path):  # 21
    runs, root = tmp_path / "runs", tmp_path / "ck"
    root.mkdir()
    shutil.copytree(tiny_root / "vae-v1", root / "vae-v1")
    common = ["--data", str(tiny_data), "--cpu", "--root", str(root), "--eval-windows", "16"]
    tiny = ["--runs", str(runs), "--arch", json.dumps({"d": 64, "depth": 2}), "--batch", "4"]
    assert P.main(["train", "--steps", "5", "--name", "cli", *tiny, *common]) == 1  # 5 steps cannot pass
    (run,) = list(runs.iterdir())
    assert P.main(["eval", str(run), *common, "--json", str(tmp_path / "e.json")]) == 1
    assert json.loads((tmp_path / "e.json").read_text())["verdict"] == "FAIL"
    assert P.main(["promote", str(run), "--name", "x", "--root", str(root)]) == 1
    assert P.main(["promote", str(run), "--name", "x", "--force", "--root", str(root)]) == 0
    assert P.main(["eval", "x", *common]) == 1

    assert P.main(["train", "--steps", "2", "--name", "reg", "--set", "objective=regression",
                   "--set", "video_stream=false", *tiny, *common]) == 1
    (reg,) = [p for p in runs.iterdir() if p.name.endswith("-reg")]
    cfg = json.loads((reg / "config.json").read_text())["model"]
    assert cfg["objective"] == "regression" and cfg["video_stream"] is False
    assert (reg / "metrics.json").exists()

    assert P.main(["train", "--set", "objective=regression", *tiny, *common]) == 2
    assert P.main(["train", "--set", "nope=1", *tiny, *common]) == 2
    assert P.main(["train", "--set", "horizon", *tiny, *common]) == 2

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert P.main(["show", str(run), *common]) == 2
        pytest.skip("matplotlib not installed; show exits 2")
    out = tmp_path / "s.png"
    assert P.main(["show", "x", *common, "--out", str(out)]) == 0
    assert out.stat().st_size > 1000


@pytest.mark.skipif(not (REPO / "data/step3-v1").exists(), reason="needs data/step3-v1")
def test_real_fork_references():  # 22
    ds = open_dataset(REPO / "data/step3-v1", check=False)
    ref = E.fork_references(ds)
    assert np.isfinite(ref[1:]).all()
    w = ds.split("val").windows(future_frames=())
    w = w.subset(w.approaches_section)
    info = E.fork_info(w)
    m = (info.entry >= 1) & (info.entry <= E.MAX_ENTRY)
    e, steer, pb = info.entry[m], info.steer[m], info.planned[m]
    cls, r = E.classify(steer, ref[e, 0], ref[e, 1])
    assert (cls == pb).mean() == 1.0
    assert (r < E.COMMIT_R).mean() >= 0.999
    _, r_avg = E.classify(0.5 * (ref[e, 0] + ref[e, 1]), ref[e, 0], ref[e, 1])
    assert (r_avg < E.COMMIT_R).mean() == 0.0
    sep = np.sqrt(((ref[:, 0] - ref[:, 1]) ** 2).mean(-1))  # prototypes/out/fork_metric.json: 1.617 … 0.852
    assert abs(sep[1] - 1.617) < 0.01 and abs(sep[12] - 0.852) < 0.01
