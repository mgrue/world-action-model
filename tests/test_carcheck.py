"""Step 4 car-visibility check (SPEC §4, §10 tests 13–15)."""

import math

import numpy as np
import pytest

from toycar import State, render
from toycar.dataset import open_dataset
from wam import carcheck as cc

N = 256


@pytest.fixture(scope="module")
def val(tiny_data):
    es = open_dataset(tiny_data, check=False).split("val")
    frames = cc.eval_frames(es, N)
    states = [State(*map(float, es.arrays["states"][e, t])) for e, t in frames]
    rps = [es.render_params(e) for e, _ in frames]
    orig = np.stack([render(s, es.track, rp) for s, rp in zip(states, rps)])
    branch = [int(es.arrays["branch"][e, t]) for e, t in frames]
    return es, frames, states, rps, orig, branch


def test_eval_frames_fixed(val):
    es, frames, *_ = val
    assert len(frames) == N and len(set(frames)) == N
    assert frames == cc.eval_frames(es, N)
    per_ep = np.bincount([e for e, _ in frames], minlength=len(es))
    assert per_ep.max() - per_ep.min() <= 1
    assert cc.eval_frames(es, 3) == [(e, t) for e, t in cc.eval_frames(es, 3)] and len(cc.eval_frames(es, 3)) == 3


def test_estimator_on_originals(val):  # test 13
    es, frames, states, rps, orig, _ = val
    pe, he = [], []
    for s, rp, f in zip(states, rps, orig):
        po = cc.estimate(f, cc.background(es.track, rp), rp)
        assert po is not None and po[2] is not None
        pe.append(math.hypot(po[0] - s.x, po[1] - s.y))
        he.append(cc._wrap_deg(po[2] - s.theta))
    assert np.percentile(pe, 95) <= 0.2
    assert np.percentile(he, 95) <= 10.0


def _fakes(s, track, rp, orig, bg, rng):
    return {
        "copy": orig.copy(),
        "noise": np.clip(orig.astype(int) + rng.integers(-4, 5, orig.shape), 0, 255).astype(np.uint8),
        "shift": render(State(s.x + 1 / cc.PX, s.y, s.theta, s.v), track, rp),
        "erase": bg.copy(),
        "backwards": render(State(s.x, s.y, s.theta + math.pi, s.v), track, rp),
        "block": np.round(np.repeat(np.repeat(orig.reshape(8, 8, 8, 8, 3).mean(axis=(1, 3)), 8, 0), 8, 1)).astype(np.uint8),
    }


def test_fakes_keep_verdicts(val):  # test 14
    es, frames, states, rps, orig, branch = val
    rng = np.random.default_rng(0)
    recs: dict[str, list] = {}
    for s, rp, o in zip(states, rps, orig):
        for k, v in _fakes(s, es.track, rp, o, cc.background(es.track, rp), rng).items():
            recs.setdefault(k, []).append(v)
    expect = {"copy": "PASS", "noise": "PASS", "shift": "FAIL", "erase": "FAIL", "backwards": "FAIL", "block": "FAIL"}
    for k, want in expect.items():
        res = cc.check(orig, np.stack(recs[k]), states, rps, es.track, branch=branch)
        assert res["all"]["verdict"] == want, (k, res["all"])
        assert len(res["rows"]) == N
    copy = cc.check(orig, orig, states, rps, es.track, branch=branch)
    assert copy["all"]["pos_max"] == 0 and copy["all"]["head_max"] == 0 and copy["all"]["psnr"] == 99.0
    erased = cc.check(orig, np.stack(recs["erase"]), states, rps, es.track, branch=branch)
    assert erased["all"]["found"] == 0.0


def _row(readable=True, pos=0.05, head=2.0):
    return {"found": True, "pos": pos, "head": head, "pos_true": 0.05, "head_true": 2.0, "psnr": 40.0,
            "psnr_car": 35.0, "psnr_bg": 45.0, "section": False, "fast": False, "kerb": False, "readable": readable}


def test_unreadable_rows():  # test 15
    rows = [_row() for _ in range(199)] + [_row(readable=False, pos=5.0, head=180.0)]
    s = cc.summarise(rows)["all"]
    assert s["n"] == 200 and s["skip"] == 1 and s["verdict"] == "PASS"  # left out, not failing
    assert s["pos_max"] == 0.05 and s["head_max"] == 2.0
    rows = [_row() for _ in range(196)] + [_row(readable=False) for _ in range(4)]  # 2% unreadable
    s = cc.summarise(rows)["all"]
    assert s["skip"] == 4 and s["verdict"] == "INVALID"
    rows = [_row() for _ in range(100)] + [_row(pos=0.7)]
    assert cc.summarise(rows)["all"]["verdict"] == "FAIL"


def test_slices_and_json(val):
    es, frames, states, rps, orig, branch = val
    res = cc.check(orig, orig, states, rps, es.track, branch=branch, label="copy")
    js = cc.jsonable(res)
    assert "rows" not in js and set(js) <= set(cc.SLICES) and "all" in js
    if any(b != -1 for b in branch):
        assert js["section"]["n"] == sum(b != -1 for b in branch)
    with pytest.raises(ValueError):
        cc.check(orig, orig[:-1], states, rps, es.track, branch=branch)


def test_contact_sheet(val, tmp_path):
    pytest.importorskip("matplotlib")
    es, frames, states, rps, orig, branch = val
    res = cc.check(orig, orig, states, rps, es.track, branch=branch)
    out = tmp_path / "c.png"
    cc.contact_sheet(out, "copy", (res["rows"], orig, orig), {"x": (res["rows"][:5], orig[:5], orig[:5])})
    assert out.stat().st_size > 1000
