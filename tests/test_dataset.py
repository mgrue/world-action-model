"""Step 3 dataset: recording, Windows, integrity checks and CLI (Step 3 SPEC §10)."""

import json
import shutil

import numpy as np
import pytest

from toycar import RenderParams, State, Track, render, step_state
from toycar import dataset as D
from toycar.expert import INSTRUCTIONS, Expert, ExpertAction

SMALL = {"train": 12, "val": 6, "test": 6}
LENGTH = 60
ARRAYS = {
    "states": ((LENGTH + 1, 4), np.float64), "clean": ((LENGTH, 2), np.float32),
    "executed": ((LENGTH, 2), np.float32), "perturbed": ((LENGTH,), np.bool_),
    "planned_branch": ((LENGTH,), np.int8), "station_m": ((LENGTH + 1,), np.float32),
    "branch": ((LENGTH + 1,), np.int8), "k": ((), np.int32), "seed": ((), np.int64),
    "instruction": ((), np.int8), "direction": ((), np.int8), "start_route": ((), np.int8),
    "length": ((), np.int32), "render_params": ((5, 3), np.uint8), "frames_sha256": ((), np.dtype("<U64")),
}


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    return D.record(tmp_path_factory.mktemp("ds") / "small", episodes=SMALL, length=LENGTH, workers=2)


@pytest.fixture(scope="module")
def ds(small):
    return D.open_dataset(small)


def _load(path, split):
    with np.load(path / f"{split}.npz", allow_pickle=False) as z:
        return {n: z[n] for n in z.files}


def test_record_layout(small):
    assert sorted(p.name for p in small.iterdir()) == ["manifest.json", "test.npz", "train.npz", "val.npz"]
    for sp in D.SPLITS:
        a = _load(small, sp)
        E = SMALL[sp]
        assert set(a) == set(ARRAYS)
        for name, (shape, dtype) in ARRAYS.items():
            assert a[name].shape == (E, *shape), name
            assert a[name].dtype == dtype, name
        assert (a["seed"] == D.SEED_BLOCKS[sp] + np.arange(E)).all()
        assert (a["k"] == np.arange(E)).all()
        assert (a["length"] == LENGTH).all()
        for k in range(E):
            ins, d = D.COMBINATIONS[k % 6]
            assert INSTRUCTIONS[a["instruction"][k]] == ins and D.DIRECTIONS[a["direction"][k]] == d
            assert (a["start_route"][k] == -1) == (ins == "any")
            if ins != "any":
                assert a["start_route"][k] == ("outer", "inner").index(ins)
    m = json.loads((small / "manifest.json").read_text())
    for key in ["format_version", "toycar_version", "python_version", "numpy_version", "platform", "created",
                "git_commit", "git_dirty", "source_sha256", "episode_ticks", "episodes", "seed_blocks",
                "combinations", "start_rule", "expert_params", "track_params", "render_defaults", "colour_jitter",
                "summary"]:
        assert key in m, key
    assert m["format_version"] == 1 and m["episode_ticks"] == LENGTH and m["episodes"] == SMALL
    assert set(m["source_sha256"]) == set(D.SOURCE_FILES)
    s = m["summary"]
    for key in ["splits", "combinations", "any_outer_fraction", "perturbed_fraction", "approaches_section_fraction"]:
        assert key in s
    assert s["splits"]["train"]["episodes"] == 12 and s["splits"]["train"]["ticks"] == 12 * LENGTH


def test_record_workers_identical(small, tmp_path):
    one = D.record(tmp_path / "one", episodes=SMALL, length=LENGTH, workers=1)
    for sp in D.SPLITS:
        a, b = _load(small, sp), _load(one, sp)
        for n in ARRAYS:
            assert np.array_equal(a[n], b[n]), (sp, n)
        assert (small / f"{sp}.npz").read_bytes() == (one / f"{sp}.npz").read_bytes()


def test_record_atomic_and_force(tmp_path, monkeypatch):
    out = tmp_path / "ds"
    tiny = {"train": 2, "val": 1, "test": 1}
    D.record(out, episodes=tiny, length=20, workers=1)
    with pytest.raises(FileExistsError):
        D.record(out, episodes=tiny, length=20, workers=1)

    real = D._record_episode
    calls = {"n": 0}

    def flaky(job):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyError("injected")
        return real(job)

    monkeypatch.setattr(D, "_record_episode", flaky)
    fresh = tmp_path / "fresh"
    with pytest.raises(KeyError):
        D.record(fresh, episodes=tiny, length=20, workers=1)
    assert not fresh.exists() and not (tmp_path / "fresh.tmp").exists()
    calls["n"] = 0
    before = (out / "train.npz").read_bytes()
    with pytest.raises(KeyError):
        D.record(out, episodes=tiny, length=20, workers=1, force=True)
    assert (out / "train.npz").read_bytes() == before and not (tmp_path / "ds.tmp").exists()

    monkeypatch.setattr(D, "_record_episode", real)
    D.record(out, episodes={"train": 3, "val": 1, "test": 1}, length=20, workers=1, force=True)
    assert len(_load(out, "train")["k"]) == 3


def test_crash_aborts(tmp_path, monkeypatch):
    def into_grass(self, state, info):
        return ExpertAction((1.0, 1.0), (1.0, 1.0), False)

    monkeypatch.setattr(Expert, "act", into_grass)
    out = tmp_path / "crash"
    with pytest.raises(RuntimeError, match=r"seed 1000000\b.*instruction outer, direction cw"):
        D.record(out, episodes={"train": 1, "val": 0, "test": 0}, length=300, workers=1)
    assert not out.exists() and not (tmp_path / "crash.tmp").exists()


def test_episode_replays(ds):
    for sp in D.SPLITS:
        es = ds.split(sp)
        for i in (0, len(es) - 1):
            ep = es.episode(i)
            res = D._record_episode((sp, ep.k, ep.length))
            for n in ("states", "clean", "executed", "perturbed", "planned_branch", "station_m", "branch"):
                assert np.array_equal(res[n], getattr(ep, n)), (sp, i, n)
            assert res["frames_sha256"] == ep.frames_sha256
            assert ep.render_params == D.sample_render_params(ep.seed)


def test_window_alignment(ds):
    es = ds.split("train")
    w = es.windows()
    track = Track.default()
    for row in np.linspace(0, len(w) - 1, 6).astype(int):
        item = w[int(row)]
        e, t = int(item["episode"]), int(item["t"])
        ep = es.episode(e)
        rp = ep.render_params
        st = [State(*map(float, r)) for r in ep.states]
        assert np.array_equal(item["obs_frame"], render(st[t], track, rp))
        assert np.array_equal(item["state"], st[t].features())
        assert np.array_equal(item["state_raw"], ep.states[t])
        assert list(item["future_index"]) == list(range(16))
        for k in range(16):
            assert np.array_equal(item["future_frames"][k], render(st[t + 1 + k], track, rp))
            nxt = step_state(st[t + k], [float(x) for x in ep.executed[t + k]])
            assert np.array_equal(item["future_frames"][k], render(nxt, track, rp))
            assert np.array_equal(item["actions"][k], ep.clean[t + k])
        assert int(item["seed"]) == ep.seed and int(item["instruction"]) == INSTRUCTIONS.index(ep.instruction)


@pytest.mark.parametrize("horizon", [16, 8])
@pytest.mark.parametrize("clean_future", [True, False])
def test_clean_future_filter(ds, horizon, clean_future):
    es = ds.split("train")
    w = es.windows(horizon=horizon, clean_future=clean_future, future_frames=())
    expect = [
        (e, t)
        for e in range(len(es))
        for t in range(LENGTH - horizon + 1)
        if not (clean_future and es.arrays["perturbed"][e, t : t + horizon].any())
    ]
    assert w.index.dtype == np.int32 and w.index.shape == (len(expect), 2)
    assert [tuple(map(int, r)) for r in w.index] == expect
    if clean_future:
        for i in range(0, len(w), 7):
            item = w[i]
            assert np.array_equal(item["executed"], item["actions"]) and not item["perturbed"].any()
    else:
        assert (es.arrays["perturbed"].any()), "fixture should contain bursts"


def test_future_frames_subset(ds):
    es = ds.split("val")
    w = es.windows(future_frames=(15, 3, 11, 7))
    item = w[0]
    assert item["future_frames"].shape == (4, 64, 64, 3)
    assert list(item["future_index"]) == [3, 7, 11, 15]
    full = es.windows()[0]
    assert np.array_equal(item["future_frames"], full["future_frames"][[3, 7, 11, 15]])
    assert es.windows(future_frames=())[0]["future_frames"].shape == (0, 64, 64, 3)
    for bad in [(16,), (-1,), (3, 3)]:
        with pytest.raises(ValueError):
            es.windows(future_frames=bad)
    for h in (0, LENGTH + 1):
        with pytest.raises(ValueError):
            es.windows(horizon=h)
    with pytest.raises(IndexError):
        w[len(w)]
    with pytest.raises(ValueError):
        ds.split("dev")


def test_section_flags(ds):
    n_appr = 0
    for sp in D.SPLITS:
        es = ds.split(sp)
        w = es.windows()
        br = es.arrays["branch"]
        for (e, t), appr, ins in zip(w.index, w.approaches_section, w.in_section):
            assert ins == (br[e, t] != -1)
            assert appr == (br[e, t] == -1 and bool((br[e, t + 1 : t + 17] != -1).any()))
        n_appr += int(w.approaches_section.sum())
        if len(w):
            item = w[int(np.argmax(w.approaches_section))]
            assert bool(item["approaches_section"]) == bool(w.approaches_section.max())
    assert n_appr > 0


def test_hash_check(small, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(small, bad)
    a = _load(bad, "val")
    a["render_params"] = a["render_params"].copy()
    a["render_params"][:, 0, 0] ^= 0x40  # every episode, so the open check hits one
    D._save_npz(bad / "val.npz", a)
    with pytest.raises(ValueError, match="split val"):
        D.open_dataset(bad)
    with pytest.raises(ValueError, match="split val"):
        D.open_dataset(bad, check=False).verify()

    fake = tmp_path / "fake"
    shutil.copytree(small, fake)
    m = json.loads((fake / "manifest.json").read_text())
    m["source_sha256"]["render.py"] = "0" * 64
    (fake / "manifest.json").write_text(json.dumps(m))
    with pytest.warns(UserWarning, match="render.py"):
        D.open_dataset(fake)

    m["format_version"] = 2
    (fake / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="format_version"):
        D.open_dataset(fake, check=False)


def test_eval_subset_fixed(small):
    a = D.open_dataset(small, check=False).split("val").eval_subset(n=40)
    b = D.open_dataset(small, check=False).split("val").eval_subset(n=40)
    assert np.array_equal(a.index, b.index)
    es = D.open_dataset(small, check=False).split("val")
    full = es.windows()
    per_ep = np.bincount(full.index[:, 0], minlength=len(es))
    E = len(es)
    quota = [40 // E + (1 if i < 40 % E else 0) for i in range(E)]
    got = np.bincount(a.index[:, 0], minlength=E)
    assert list(got) == [min(q, int(p)) for q, p in zip(quota, per_ep)]
    assert len(a) == min(40, sum(got))
    rows = {tuple(r) for r in full.index.tolist()}
    assert all(tuple(r) in rows for r in a.index.tolist())
    assert (np.diff(a.index[:, 0]) >= 0).all()
    big = es.eval_subset(n=10**6)
    assert np.array_equal(big.index, full.index)


def test_colour_jitter(tmp_path, monkeypatch):
    lo, hi = D.BRIGHTNESS_RANGE
    for seed in range(1000):
        rp = D.sample_render_params(seed)
        assert rp.supersample == 4
        for f in D.COLOUR_FIELDS:
            for c, c0 in zip(getattr(rp, f), getattr(RenderParams(), f)):
                assert 0 <= c <= 255
                assert max(0, int(np.floor(c0 * lo - D.CHANNEL_OFFSET))) <= c <= min(255, int(np.ceil(c0 * hi + D.CHANNEL_OFFSET)))
    assert D.sample_render_params(5) == D.sample_render_params(5) != D.sample_render_params(6)

    tiny = {"train": 6, "val": 1, "test": 1}
    on = D.record(tmp_path / "on", episodes=tiny, length=40, workers=1)
    monkeypatch.setattr(D, "sample_render_params", lambda seed: RenderParams())
    off = D.record(tmp_path / "off", episodes=tiny, length=40, workers=1)
    for sp in D.SPLITS:
        a, b = _load(on, sp), _load(off, sp)
        for n in ("states", "clean", "executed", "perturbed", "planned_branch", "branch", "station_m"):
            assert np.array_equal(a[n], b[n]), (sp, n)
        assert (b["render_params"] == D._rp_to_array(RenderParams())).all()
        assert not (a["render_params"] == b["render_params"]).all()


def test_probes(ds):
    es = ds.split("test")
    for name, probe in D.PROBES.items():
        if probe.track_params is not None:
            continue
        for i in range(len(es)):
            rp = es.episode(i).render_params
            pp = es.render_params(i, name)
            changed = {f for f in D.COLOUR_FIELDS if getattr(rp, f) != getattr(pp, f)}
            assert changed <= set(probe.colours), name
            for f, v in probe.colours.items():
                assert getattr(pp, f) == (getattr(rp, v) if isinstance(v, str) else v)
        fr = es.frames(0, ticks=[0, 5], probe=name)
        assert fr.shape == (2, 64, 64, 3) and not np.array_equal(fr, es.frames(0, ticks=[0, 5]))
    assert es.render_params(0, "no_kerb").kerb == es.episode(0).render_params.road
    for call in (lambda: es.windows(probe="wide_fork"), lambda: es.frames(0, probe="wide_fork"),
                 lambda: es.eval_subset(probe="wide_fork"), lambda: D.frame_dataset(es, probe="wide_fork"),
                 lambda: es.windows(probe="nope")):
        with pytest.raises(ValueError):
            call()
    w = es.windows(probe="brown_grass")
    assert tuple(w[0]["obs_frame"][0, 0]) != tuple(es.windows()[0]["obs_frame"][0, 0])


def test_frame_dataset(ds):
    es = ds.split("val")
    fd = D.frame_dataset(es)
    assert len(fd) == len(es) * (LENGTH + 1)
    for e, t in [(0, 0), (0, LENGTH), (1, 0), (3, 17), (len(es) - 1, LENGTH)]:
        assert np.array_equal(fd[e * (LENGTH + 1) + t], es.frames(e, ticks=[t])[0])
        assert fd.locate(e * (LENGTH + 1) + t) == (e, t)
    assert fd.locate(-1) == (len(es) - 1, LENGTH)
    with pytest.raises(IndexError):
        fd[len(fd)]
    with pytest.raises(IndexError):
        es.episode(len(es))


def test_dataloader_compat(ds):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    w = ds.split("val").windows()
    batch = next(iter(DataLoader(w, batch_size=4, num_workers=0)))
    assert batch["obs_frame"].shape == (4, 64, 64, 3) and batch["obs_frame"].dtype == torch.uint8
    assert batch["future_frames"].shape == (4, 16, 64, 64, 3)
    assert batch["actions"].shape == (4, 16, 2) and batch["state"].shape == (4, 5)
    assert batch["instruction"].shape == (4,) and batch["approaches_section"].dtype == torch.bool


def test_cli(tmp_path, capsys):
    out = tmp_path / "cli"
    assert D.main(["record", "--out", str(out), "--episodes", "train=6,val=6,test=1", "--length", "40", "--workers", "1"]) == 0
    assert D.main(["record", "--out", str(out), "--episodes", "train=6,val=6,test=1", "--length", "40"]) == 1
    assert D.main(["verify", str(out)]) == 0
    assert "PASS" in capsys.readouterr().out
    assert D.main(["stats", str(out)]) == 0
    assert "speed histogram" in capsys.readouterr().out
    pytest.importorskip("matplotlib")
    png = tmp_path / "w.png"
    assert D.main(["plot", str(out), "--seed", "3", "--out", str(png)]) == 0
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    png2 = tmp_path / "w2.png"
    assert D.main(["plot", str(out), "--episode", "0", "--t", "5", "--probe", "brown_grass", "--out", str(png2)]) == 0
    assert png2.exists()
    png3 = tmp_path / "w3.png"
    assert D.main(["plot", str(out), "--seed", "1", "--approaching", "--zoom", "12", "--out", str(png3)]) == 0
    assert png3.exists()
    with pytest.raises(SystemExit):
        D.main(["plot", str(out), "--zoom", "0"])


def test_zoom_box():
    # a still car mid-track: exactly zoom_m wide (12 m = 19.2 → 20 px), centred on the car
    r0, r1, c0, c1 = D._zoom_box(np.array([[20.0, 20.0]]), 12.0)
    assert r1 - r0 == c1 - c0 == 20 and (r0 + r1) / 2 == (c0 + c1) / 2 == 32
    # a long path grows the box to keep it (plus a car length each side) in view
    path = np.array([[10.0, 20.0], [22.0, 21.0]])
    r0, r1, c0, c1 = D._zoom_box(path, 6.0)
    assert c1 - c0 >= (12.0 + 2 * 4.4) * 1.6 and c0 <= 10 * 1.6 and c1 >= 22 * 1.6
    # near the edge the box is shifted inside the Frame, never shrunk
    r0, r1, c0, c1 = D._zoom_box(np.array([[1.0, 39.0]]), 12.0)
    assert (c0, r1) == (0, 64) and r1 - r0 == c1 - c0 == 20
    assert D._zoom_box(np.array([[20.0, 20.0]]), 40.0) == (0, 64, 0, 64)
