"""Step 9 closed loop: Evaluation sets, metrics, gates, Saboteurs, CLI and Run directory (SPEC §12)."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toycar.dataset import COMBINATIONS  # noqa: E402
from toycar.expert import EpisodeScore  # noqa: E402
from wam import closedloop as CL  # noqa: E402
from wam import drive as D  # noqa: E402
from wam import model as M  # noqa: E402
from wam.train import compressor_card_sha256  # noqa: E402

TINY = {"d": 64, "depth": 2}


def test_eval_sets():
    base = 9_000_000
    g, r, d = CL.episode_specs("gate", 120), CL.episode_specs("robustness", 60), CL.episode_specs("dev", 60)
    assert [s.seed for s in g] == [base + 100_000 + k for k in range(120)]
    assert [s.seed for s in r] == [base + 200_000 + k for k in range(60)]
    assert [s.seed for s in d] == [base + 10_000 + k for k in range(60)]
    assert not ({s.seed for s in g} & {s.seed for s in r} | {s.seed for s in g} & {s.seed for s in d})
    for k, s in enumerate(g):
        assert (s.instruction, s.direction) == COMBINATIONS[k % 6] and s.condition == "none"
    for n in range(6, 121, 6):
        counts = {c: sum((s.instruction, s.direction) == c for s in g[:n]) for c in COMBINATIONS}
        assert set(counts.values()) == {n // 6}
    assert all(s.condition == "push" for s in CL.episode_specs("robustness", 6, "push"))
    for bad in ((("gate", 7)), ("gate", 126), ("robustness", 66), ("nope", 6)):
        with pytest.raises(ValueError):
            CL.episode_specs(*bad)
    with pytest.raises(ValueError):
        CL.episode_specs("gate", 6, "rain")


def _total(**kw):
    t = {"crash_rate": 0.0, "crash_rate_ci": (0, 0.03), "frac_laps_mean": 4.0, "instruction_followed_rate": 1.0,
         "instruction_followed_ci": (0.97, 1), "outer_fraction": 0.5, "outer_fraction_ci": (0.4, 0.6)}
    t.update(kw)
    return t


def test_gates():
    ok = CL.gates(_total())
    assert all(g["pass"] for g in ok.values())
    edge = CL.gates(_total(crash_rate=0.05, frac_laps_mean=3.5, instruction_followed_rate=0.95, outer_fraction=0.30))
    assert all(g["pass"] for g in edge.values())
    assert CL.gates(_total(outer_fraction=0.70))["balance"]["pass"]
    bad = CL.gates(_total(crash_rate=0.0501, frac_laps_mean=3.4999, instruction_followed_rate=0.9499,
                          outer_fraction=0.7001))
    assert not any(g["pass"] for g in bad.values())
    assert not CL.gates(_total(outer_fraction=0.2999))["balance"]["pass"]
    assert not CL.gates(_total(outer_fraction=None))["balance"]["pass"]

    kw = dict(driver="wam-v1", is_policy=True, scheme=D.DEFAULT_SCHEME, set_name="gate", n=120, condition="none",
              mode="gate")
    assert CL.verdict(ok, **kw) == ("PASS", None)
    assert CL.verdict(bad, **kw) == ("FAIL", None)
    assert CL.verdict(ok, **{**kw, "driver": "wam-v1-novideo"}) == (None, "not gated")
    assert CL.verdict(ok, **{**kw, "driver": "wam-v1", "is_policy": False}) == (None, "not gated")
    assert CL.verdict(ok, **{**kw, "driver": "expert", "is_policy": False}) == (None, "not gated")
    assert CL.verdict(ok, **{**kw, "scheme": replace(D.DEFAULT_SCHEME, execute_k=2)}) == (None, "non-default scheme")
    assert CL.verdict(ok, **{**kw, "n": 114}) == (None, "not a gate run")
    assert CL.verdict(ok, **{**kw, "condition": "push"}) == (None, "not a gate run")
    assert CL.verdict(ok, **{**kw, "set_name": "dev", "mode": "sweep:replan"}) == (None, "not a gate run")


SABOTEUR_FAILS = {
    "zero-steer": {"crashes", "laps", "balance"},
    "averaged-fork": {"crashes", "laps", "balance"},
    "slow": {"laps"},
    "wrong-branch": {"followed"},
    "always-outer": {"balance"},
}


def test_saboteurs():
    specs = CL.episode_specs("gate", 24)
    g = CL.gates(CL.metrics(D.run_episodes(D.ExpertDriver(), specs, shadow_expert=False))["total"])
    assert all(x["pass"] for x in g.values()), g
    for kind, want in SABOTEUR_FAILS.items():
        m = CL.metrics(D.run_episodes(D.SaboteurDriver(kind), specs, shadow_expert=False))
        g = CL.gates(m["total"])
        assert {k for k, x in g.items() if not x["pass"]} == want, (kind, g)
        if kind == "averaged-fork":
            assert m["total"]["island_crashes"] == m["total"]["crashes"] > 0
    with pytest.raises(ValueError):
        D.SaboteurDriver("drunk")


def _rec(spec, dist, *, crash_tick=None, branch=None, replans=(), push_ticks=(), ad=None, ex=None, dist0=0.0):
    T = len(dist)
    score = EpisodeScore(spec.instruction, spec.direction, T, "crash" if crash_tick else "max_steps",
                         bool(crash_tick), 0, 10.0, 0, (), 0, 0, True, not crash_tick)
    z = np.zeros(T + 1)
    return D.EpisodeRecord(
        spec=spec, score=score, fractional_laps=0.5, crash_tick=crash_tick, crash_station_m=3.0 if crash_tick else None,
        crash_in_section=False if crash_tick else None, push_ticks=list(push_ticks), x=z, y=z, theta=z, v=z,
        branch=np.full(T + 1, -1, np.int8) if branch is None else np.asarray(branch, np.int8), station_m=z,
        action_driver=np.zeros((T, 2)) if ad is None else ad, action_exec=np.zeros((T, 2)), dist_m=np.asarray(dist, float),
        expert_steer=np.zeros(T) if ex is None else ex, selected=np.full(T, -1), dist0_m=dist0, replans=list(replans))


def test_metrics():
    lo, hi = CL.wilson(0, 120)
    assert lo == 0 and abs(hi - 0.031) < 5e-4
    lo, hi = CL.wilson(30, 120)
    assert abs(lo - 0.181) < 5e-4 and abs(hi - 0.334) < 5e-4
    lo, hi = CL.wilson(6, 120)
    assert abs(lo - 0.023) < 5e-4 and abs(hi - 0.105) < 5e-4

    sp = D.EpisodeSpec(1, "outer", "cw")
    dist = np.linspace(0, 1.9, 20)
    m = CL.group_metrics([_rec(sp, dist)])
    assert math.isclose(m["dist_mean_m"], dist.mean()) and math.isclose(m["dist_max_m"], 1.9)
    assert math.isclose(m["dist_p95_m"], np.percentile(dist, 95))

    # action error: bins by the State's distance (dist0 then dist_m[:-1]); `any` Ticks near the Fork excluded
    ad = np.zeros((4, 2))
    ad[:, 0] = [0.1, 0.2, 0.3, 0.4]
    r = _rec(sp, [0.4, 0.7, 1.2, 0.0], ad=ad, dist0=0.1)
    ae = CL.action_error([r])
    assert ae["0-0.3"]["n"] == 1 and math.isclose(ae["0-0.3"]["rmse"], 0.1)
    assert ae["0.3-0.6"]["n"] == 1 and math.isclose(ae["0.3-0.6"]["rmse"], 0.2)
    assert ae[">1"]["n"] == 1 and math.isclose(ae[">1"]["rmse"], 0.4)
    T = 30
    br = np.full(T + 1, -1)
    br[20:] = 0
    ra = _rec(D.EpisodeSpec(2, "any", "cw"), np.zeros(T), branch=br)
    assert CL.action_error([ra])["all"]["n"] == 20 - 12  # Ticks 8 … 19 and the section are excluded

    # Fork flips: two approaches, one consistent and one that flips; ccw steer is mirrored
    H = 16
    ref = np.full((17, 2, H), np.nan)
    ref[1:, 0], ref[1:, 1] = 0.5, -0.5
    T = 40
    br = np.full(T + 1, -1)
    br[20:25] = 0
    br[35:] = 1
    outer, inner = np.full(H, 0.5), np.full(H, -0.5)
    reps = [(t, outer) for t in range(10, 20)] + [(t, outer if t < 30 else inner) for t in range(25, 35)]
    ff = CL.fork_flips([_rec(D.EpisodeSpec(3, "any", "cw"), np.zeros(T), branch=br, replans=reps)], ref)
    assert (ff["approaches"], ff["flips"], ff["any_approaches"], ff["any_flips"]) == (2, 1, 2, 1)
    assert ff["fraction_uncommitted"] == 0
    mirrored = [(t, -s) for t, s in reps]
    ff2 = CL.fork_flips([_rec(D.EpisodeSpec(3, "outer", "ccw"), np.zeros(T), branch=br, replans=mirrored)], ref)
    assert (ff2["approaches"], ff2["flips"], ff2["any_approaches"]) == (2, 1, 0)

    # Push recovery: Push at 60 → peak, back within 0.3 m at Tick 72 (4 Ticks after its end at 68)
    T = 130
    d = np.full(T, 0.1)
    d[59:71] = 0.8
    d[64] = 1.4
    r = _rec(D.EpisodeSpec(4, "outer", "cw", "push"), d, push_ticks=[60, 120])
    p = CL.push_recovery([r])
    assert p["pushes"] == 2 and p["crash_within_45"] == 0
    assert p["recovery_ticks_median"] == 0.0 or p["recovered_fraction"] == 1.0
    one = CL.push_recovery([_rec(D.EpisodeSpec(4, "outer", "cw", "push"), d, push_ticks=[60])])
    assert one["recovery_ticks_median"] == 4 and math.isclose(one["peak_dist_m_mean"], 1.4)
    crash = _rec(D.EpisodeSpec(4, "outer", "cw", "push"), np.full(80, 1.0), push_ticks=[60], crash_tick=80)
    pc = CL.push_recovery([crash])
    assert pc["crash_within_45"] == 1 and pc["recovered_fraction"] == 0

    # offset_start: crash rate where the expert survives
    off = [_rec(D.EpisodeSpec(s, "outer", "cw", "offset_start"), [0.5, 0.2], crash_tick=c, dist0=1.2)
           for s, c in ((1, None), (2, 2), (3, 2))]
    refr = [_rec(D.EpisodeSpec(s, "outer", "cw", "offset_start"), [0.1], crash_tick=c) for s, c in
            ((1, None), (2, None), (3, 1))]
    o = CL.offset_recovery(off, refr)
    assert o["expert_survived"] == 2 and o["crash_rate_where_expert_survives"] == 0.5
    assert o["ticks_to_line_median"] == 2


@pytest.fixture(scope="module")
def tiny_run(tiny_root, tmp_path_factory):
    """A Run directory (checkpoint.pt + config.json) holding a tiny, untrained WAM."""
    run = tmp_path_factory.mktemp("wamrun") / "20260101-0000-tiny"
    run.mkdir()
    cfg = M.ModelConfig(**TINY)
    torch.manual_seed(0)
    m = M.WAM(cfg)
    torch.save({"state_dict": m.state_dict(), "config": cfg.to_dict(), "compressor": "vae-v1",
                "compressor_card_sha256": compressor_card_sha256("vae-v1", tiny_root)}, run / "checkpoint.pt")
    (run / "config.json").write_text("{}")
    return run


@pytest.fixture(scope="module")
def cli_run(tiny_run, tiny_root, tiny_data, tmp_path_factory):
    """`eval <tiny run>` on 6 gate episodes; also captures the Frames the Driver saw."""
    out = tmp_path_factory.mktemp("drive")
    seen: dict[int, list] = {}
    orig = D.PolicyDriver.act

    def spy(self, rows, frames, states, infos):
        for r, j in enumerate(rows):
            seen.setdefault(j, []).append(frames[r].copy())
        return orig(self, rows, frames, states, infos)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(D.PolicyDriver, "act", spy)
        code = CL.main(["eval", str(tiny_run), "--episodes", "6", "--device", "cpu", "--root", str(tiny_root),
                        "--runs", str(out), "--data", str(tiny_data), "--max-steps", "40"])
    runs = list(out.iterdir())
    return code, runs, seen


def _abs_paths(obj):
    if isinstance(obj, dict):
        return [p for v in obj.values() for p in _abs_paths(v)]
    if isinstance(obj, list):
        return [p for v in obj for p in _abs_paths(v)]
    return [obj] if isinstance(obj, str) and (obj.startswith("/") or ":\\" in obj) else []


def test_cli_run_dir(cli_run, tiny_run, tiny_root, tmp_path):
    code, runs, _ = cli_run
    assert code == 0 and len(runs) == 1
    run = runs[0]
    assert run.name.endswith("-20260101-0000-tiny")
    for f in ("config.json", "episodes.jsonl", "ticks.npz", "metrics.json", "report.md", "stdout.txt"):
        assert (run / f).exists(), f
    report = (run / "report.md").read_text()
    assert "not gated" in report
    m = json.loads((run / "metrics.json").read_text())
    assert m["verdict"] is None and m["no_verdict_reason"] == "not gated"
    assert [(r["role"], r["point"]) for r in m["rows"]] == [("driver", "default"), ("reference", "default")]
    assert all(set(r["gates"]) == {"crashes", "laps", "followed", "balance"} for r in m["rows"])
    cfg = json.loads((run / "config.json").read_text())
    lines = [json.loads(x) for x in (run / "episodes.jsonl").read_text().splitlines()]
    assert len(lines) == 12 and [ln["index"] for ln in lines] == list(range(12))
    assert _abs_paths(cfg) == [] and _abs_paths(m) == [] and _abs_paths(lines) == []
    assert cfg["scheme"] == {**D.DEFAULT_SCHEME.__dict__} and cfg["episodes"] == 6
    z = np.load(run / "ticks.npz")
    assert len(z["offsets"]) == 12 and z["x"].shape[0] == sum(ln["steps"] + 1 for ln in lines)

    base = ["--device", "cpu", "--root", str(tiny_root), "--runs", str(tmp_path)]
    assert CL.main(["eval", str(tiny_run), "--episodes", "5", *base]) == 2
    assert CL.main(["eval", "saboteur:nope", *base]) == 2
    assert CL.main(["eval", str(tmp_path / "missing"), *base]) == 2
    assert CL.main(["eval", "expert", "--sweep", "replan", "--robustness", *base]) == 2
    assert CL.main(["eval", "expert", "--robustness", "rain", *base]) == 2
    assert CL.main(["eval", "expert", "--execute-k", "0", *base]) == 2
    assert CL.main(["eval", "expert", "--bogus", *base]) == 2
    assert list(tmp_path.iterdir()) == []


def test_cli_saboteur_robustness(tmp_path):
    base = ["--device", "cpu", "--runs", str(tmp_path), "--max-steps", "150"]
    assert CL.main(["eval", "saboteur:zero-steer", "--episodes", "6", *base]) == 0
    assert CL.main(["eval", "expert", "--robustness", "none", "push", "--episodes-override", "6", *base]) == 0
    run = next(p for p in tmp_path.iterdir() if p.name.endswith("expert-robustness"))
    m = json.loads((run / "metrics.json").read_text())
    assert [(r["role"], r["condition"]) for r in m["rows"]] == [("driver", "none"), ("reference", "none"),
                                                                ("driver", "push"), ("reference", "push")]
    assert m["rows"][2]["metrics"]["push"]["pushes"] == 6 * 2
    assert "Δ crash vs none" in (run / "report.md").read_text()


def test_video_rerender(cli_run, tmp_path):
    _, runs, seen = cli_run
    run = runs[0]
    for k in (0, 3):
        frames = CL.rerender(run, k)
        ln, arr = CL.load_episode(run, k)
        assert frames.shape == (ln["steps"] + 1, 64, 64, 3)
        assert np.array_equal(frames[:ln["steps"]], np.stack(seen[k]))
    assert CL.main(["video", str(run), "--episode", "1", "--out", str(tmp_path / "v")]) == 0
    pngs = sorted((tmp_path / "v").glob("frame_*.png"))
    assert pngs and pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert CL.main(["video", str(run), "--episode", "99"]) == 2
