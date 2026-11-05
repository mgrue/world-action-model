"""Step 9 closed loop: Drivers, Scheme, noise entry, Stress conditions (`.scratch/step9-closed-loop/SPEC.md` §12)."""

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toycar.dataset import PROBES, sample_render_params  # noqa: E402
from toycar.env import CarEnv  # noqa: E402
from toycar.expert import ExpertParams, run_episode  # noqa: E402
from wam import compressor as C  # noqa: E402
from wam import drive as D  # noqa: E402
from wam import flow as Fl  # noqa: E402
from wam import model as M  # noqa: E402
from wam import policy as P  # noqa: E402
from wam.closedloop import episode_specs  # noqa: E402
from wam.policy import ActResult  # noqa: E402

TINY = {"d": 64, "depth": 2}


def tiny_policy(root, **kw):
    cfg = M.ModelConfig(**TINY, **kw)
    torch.manual_seed(0)
    m = M.WAM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    return P.Policy(m, C.load("vae-v1", "cpu", root=root))


def same_records(a, b) -> bool:
    for ra, rb in zip(a, b, strict=True):
        if ra.score != rb.score or ra.push_ticks != rb.push_ticks or ra.crash_tick != rb.crash_tick:
            return False
        for f in ("x", "y", "theta", "v", "action_driver", "action_exec", "dist_m", "selected", "branch"):
            if not np.array_equal(getattr(ra, f), getattr(rb, f)):
                return False
        if [t for t, _ in ra.replans] != [t for t, _ in rb.replans]:
            return False
        if not all(np.array_equal(x, y) for (_, x), (_, y) in zip(ra.replans, rb.replans)):
            return False
    return True


def test_expert_parity():
    specs = episode_specs("gate", 12)
    for params in (ExpertParams(), D.CLEAN):
        recs = D.run_episodes(D.ExpertDriver(params), specs, shadow_expert=False, max_steps=300)
        for sp, r in zip(specs, recs):
            want, _ = run_episode(sp.seed, sp.instruction, sp.direction, params, env=CarEnv(max_steps=300))
            assert r.score == want


def test_batch_independence(tiny_root):
    pol = tiny_policy(tiny_root)
    specs = episode_specs("dev", 6)
    a = D.run_episodes(D.PolicyDriver(pol), specs, batch=6, max_steps=24)
    b = D.run_episodes(D.PolicyDriver(pol), specs, batch=1, max_steps=24)
    c = D.run_episodes(D.PolicyDriver(pol), specs, batch=6, max_steps=24)
    assert same_records(a, c)  # deterministic
    # Noise is per row, so batching changes nothing but float rounding in batched matmuls (≈ 1e-7)
    for ra, rb in zip(a, b):
        assert replace(ra.score, progress_m=0.0) == replace(rb.score, progress_m=0.0)
        assert math.isclose(ra.score.progress_m, rb.score.progress_m, abs_tol=1e-5)
        assert [t for t, _ in ra.replans] == [t for t, _ in rb.replans]
        assert (ra.selected == rb.selected).all()
        assert np.allclose(ra.action_driver, rb.action_driver, atol=1e-5)
        assert np.allclose(ra.x, rb.x, atol=1e-5) and np.allclose(ra.y, rb.y, atol=1e-5)
    assert all(len(r.replans) == r.steps for r in a)  # default execute_k = 1
    assert all(r.ms_per_replan is not None and np.isfinite(r.expert_steer).all() for r in a)


class StubPolicy:
    """Counts `act` calls; returns zero Chunks of shape (B, N, H, 2)."""

    def __init__(self, cfg):
        self.config = cfg
        self.horizon = cfg.horizon
        self.name, self.source = "stub", None
        self.calls = []

    def act(self, frame, state, instruction, *, n_samples=1, noise=None, seed=None, return_latents=False, **kw):
        B = len(frame)
        assert noise is not None and noise[1].shape == (B * n_samples, self.horizon, 2)
        self.calls.append(B)
        a = np.zeros((B, n_samples, self.horizon, 2), np.float32)
        a[..., 1] = 0.3  # some throttle, so the car moves
        return ActResult(a, None, np.zeros(0))


@pytest.mark.parametrize("k", [1, 4])
def test_scheme_replan_ticks(k):
    stub = StubPolicy(M.ModelConfig(**TINY, video_stream=False))
    specs = episode_specs("dev", 6)
    recs = D.run_episodes(D.PolicyDriver(stub, D.Scheme(execute_k=k)), specs, max_steps=20)
    for r in recs:
        assert [t for t, _ in r.replans] == list(range(0, r.steps, k))
        assert ((r.selected >= 0) == np.isin(np.arange(r.steps), np.arange(0, r.steps, k))).all()
    assert sum(stub.calls) == sum(len(r.replans) for r in recs)
    with pytest.raises(ValueError):
        D.PolicyDriver(stub, D.Scheme(execute_k=17))
    with pytest.raises(ValueError):
        D.Scheme(execute_k=0)
    with pytest.raises(ValueError):
        D.Scheme(select="mean")


def test_coherence_select():
    H, k, rho = 16, 4, 0.9
    prev = np.zeros((H, 2))
    prev[k:, 0] = 1.0  # the overlap the next Chunk should continue
    c = np.zeros((3, H, 2))
    c[1, :, 0] = 1.0  # perfect continuation
    c[2, :, 0] = 1.0
    c[2, 0, 0] = 0.0  # off at τ = 0 (weight 1)
    assert D.select_chunk(c, prev, k, "coherence", rho) == 1
    # ρ weighting: an error at τ = 0 costs more than the same error at τ = 5
    c2 = np.ones((2, H, 2))
    c2[..., 1] = 0
    c2[0, 0, 0] = 0.0
    c2[1, 5, 0] = 0.0
    assert D.select_chunk(c2, prev, k, "coherence", rho) == 1
    # only the overlap H − k counts: differences after it are ignored
    c3 = np.ones((2, H, 2))
    c3[..., 1] = 0
    c3[0, H - k:, 0] = 5.0
    c3[1, 0, 0] = 0.9
    assert D.select_chunk(c3, prev, k, "coherence", rho) == 0
    assert D.select_chunk(np.ones((3, H, 2)), prev, k, "coherence", rho) == 0  # ties → lowest index
    assert D.select_chunk(c, None, k, "coherence", rho) == 0  # first replan → sample 0
    assert D.select_chunk(c, prev, k, "first", rho) == 0
    assert D.select_chunk(c, prev, H, "coherence", rho) == 0  # no overlap


def test_regression_single_sample(tiny_root):
    pol = tiny_policy(tiny_root, video_stream=False, objective="regression")
    drv = D.PolicyDriver(pol, D.Scheme(n_samples=4, cfg_action=2.0), keep_candidates=True)
    assert drv.scheme.n_samples == 1 and drv.scheme.cfg_action == 1.0
    recs = D.run_episodes(drv, episode_specs("dev", 6), max_steps=5)
    assert all(len(c) == 1 for c in drv.last_candidates.values())
    assert all((r.selected == 0).all() for r in recs)


def test_noise_entry(tiny_root):
    pol = tiny_policy(tiny_root)
    rng = np.random.default_rng(0)
    fr = rng.integers(0, 256, (2, 64, 64, 3), dtype=np.uint8)
    st = rng.standard_normal((2, 5)).astype(np.float32)
    noise = Fl.initial_noise(pol.config, 6, torch.Generator().manual_seed(11))
    a = pol.act(fr, st, ["outer", "any"], n_samples=3, n_steps=3, seed=11).actions
    b = pol.act(fr, st, ["outer", "any"], n_samples=3, n_steps=3, noise=noise).actions
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        pol.act(fr, st, "any", n_samples=3, seed=11, noise=noise)
    with pytest.raises(ValueError):
        pol.act(fr, st, "any", n_samples=2, noise=noise)  # 4 rows, noise for 6
    obs = pol.compressor.encode(torch.from_numpy(fr))
    sti, ins = torch.from_numpy(st), torch.tensor([0, 2])
    nz = Fl.initial_noise(pol.config, 2, torch.Generator().manual_seed(3))
    z1, a1 = Fl.sample(pol.model, pol.config, obs, sti, ins, n_steps=2, generator=torch.Generator().manual_seed(3))
    z2, a2 = Fl.sample(pol.model, pol.config, obs, sti, ins, n_steps=2, noise=nz)
    assert torch.equal(a1, a2) and torch.equal(z1, z2)
    with pytest.raises(ValueError):
        Fl.sample(pol.model, pol.config, obs, sti, ins, generator=torch.Generator(), noise=nz)
    with pytest.raises(ValueError):
        Fl.sample(pol.model, pol.config, obs, sti, ins, noise=(nz[0], nz[1][:, :8]))
    with pytest.raises(ValueError):
        Fl.sample(pol.model, pol.config, obs, sti, ins, noise=(None, nz[1]))


def test_conditions(monkeypatch):
    seed = 9_200_003
    sp = D.EpisodeSpec(seed, "outer", "cw", "brown_grass")
    env, _, _ = D.reset_env(sp)
    jit = sample_render_params(seed)
    assert env.render_params == PROBES["brown_grass"].apply(jit)
    assert env.render_params.grass == (120, 100, 50) and env.render_params.road == jit.road
    env, _, _ = D.reset_env(replace(sp, condition="none"))
    assert env.render_params == jit and env.track.params.bulge_m == 6.0
    env, _, _ = D.reset_env(replace(sp, condition="wide_fork"))
    assert env.track.params.bulge_m == 7.0
    with pytest.raises(ValueError):
        D.reset_env(replace(sp, condition="rain"))

    # offset_start: within ranges and deterministic
    for k, spec in enumerate(episode_specs("robustness", 12, "offset_start")):
        env, s, i = D.reset_env(spec)
        env2, s2, _ = D.reset_env(spec)
        assert s == s2
        route = spec.instruction if spec.instruction != "any" else None
        dists = [env.track.route(b).project(s.x, s.y).distance_m for b in ([route] if route else ["outer", "inner"])]
        assert any(1.0 - 1e-6 <= d <= 1.5 + 1e-6 for d in dists), dists
        assert 0.0 <= s.v <= 6.0 and i.direction == spec.direction

    # redrawn when invalid: the first draw is rejected → a different, valid start
    spec = episode_specs("robustness", 6, "offset_start")[0]
    _, s_first, _ = D.reset_env(spec)
    calls = {"n": 0}
    orig = CarEnv.reset

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("invalid start")
        return orig(self, *a, **kw)

    monkeypatch.setattr(CarEnv, "reset", flaky)
    _, s_redrawn, _ = D.reset_env(spec)
    monkeypatch.undo()
    assert calls["n"] >= 2 and s_redrawn != s_first

    # Push: exactly on Ticks [60 + 60m, 68 + 60m), signs from [seed, 4]
    specs = episode_specs("robustness", 6, "push")
    recs = D.run_episodes(D.ExpertDriver(), specs, shadow_expert=False, max_steps=200)
    for sp, r in zip(specs, recs):
        assert not r.score.crashed
        assert r.push_ticks == [60, 120, 180]
        rng = np.random.default_rng([sp.seed, 4])
        want_sign = {p: float(rng.choice([-1, 1])) for p in r.push_ticks}
        on = np.zeros(r.steps, bool)
        for p in r.push_ticks:
            on[p:p + 8] = True
        diff = r.action_exec[:, 0] - r.action_driver[:, 0]
        assert (diff[~on] == 0).all() and (r.action_exec[:, 1] == r.action_driver[:, 1]).all()
        for p in r.push_ticks:
            d = diff[p:p + 8]
            unclipped = np.abs(r.action_driver[p:p + 8, 0] + want_sign[p] * 0.4) <= 1
            assert np.allclose(d[unclipped], want_sign[p] * 0.4)
    strong = D.run_episodes(D.ExpertDriver(), specs[:1], shadow_expert=False, max_steps=70)[0]
    d = strong.action_exec[60:68, 0] - strong.action_driver[60:68, 0]
    assert np.isclose(np.abs(d).max(), 0.4)
    st2 = D.run_episodes(D.ExpertDriver(), [replace(specs[0], condition="push_strong")], shadow_expert=False,
                         max_steps=70)[0]
    assert np.isclose(np.abs(st2.action_exec[60:68, 0] - st2.action_driver[60:68, 0]).max(), 0.5)
    assert math.isclose(D.PUSH_AMP["push_strong"], 0.5)
