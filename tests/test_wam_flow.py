"""Steps 5–8 flow matching (SPEC §12 tests 8–12)."""

import pytest

torch = pytest.importorskip("torch")

from wam import flow as Fl  # noqa: E402
from wam.model import NULL, ModelConfig  # noqa: E402


def batch(cfg, B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"obs": torch.randn(B, 16, 8, 8, generator=g), "state": torch.randn(B, 5, generator=g),
            "instr": torch.tensor([0, 1, 2, 0])[:B],
            "fut": torch.randn(B, cfg.n_future, 16, 8, 8, generator=g) if cfg.video_stream else None,
            "actions": torch.rand(B, cfg.horizon, 2, generator=g) * 2 - 1}


class Fake:
    """Records calls; returns constant velocities (or instruction-dependent ones)."""

    def __init__(self, cfg, vz=0.3, va=-0.2, by_instr=False):
        self.cfg, self.vz, self.va, self.by_instr, self.calls = cfg, vz, va, by_instr, []

    def __call__(self, obs, state, instr, x_z, x_a, t):
        self.calls.append({"B": obs.shape[0], "instr": instr.clone(), "x_a": x_a.clone(), "t": t.clone()})
        B = obs.shape[0]
        if self.by_instr:
            va = torch.where(instr == NULL, 1.0, 3.0).view(B, 1, 1).expand_as(x_a).clone()
            vz = None if x_z is None else torch.where(instr == NULL, -1.0, 2.0).view(B, 1, 1, 1, 1).expand_as(x_z).clone()
            return vz, va
        return (None if x_z is None else torch.full_like(x_z, self.vz)), torch.full_like(x_a, self.va)


def test_noising_and_t():  # 8
    x, eps, t = torch.randn(5, 3, 2), torch.randn(5, 3, 2), torch.rand(5)
    xt, tgt = Fl.noised(x, eps, t)
    assert torch.allclose(xt, t.view(5, 1, 1) * eps + (1 - t.view(5, 1, 1)) * x)
    assert torch.equal(tgt, eps - x)
    ts = Fl.sample_t(100_000, 1.65, torch.Generator().manual_seed(0))
    assert ts.min() > 0 and ts.max() < 1
    assert abs(Fl.timeshift(1.65, 0.5) - 0.6226) < 1e-4
    assert abs(float(ts.median()) - 0.6226) < 0.005


def test_action_normalisation():  # 9
    cfg = ModelConfig()
    assert cfg.action_mean == (0.0, 0.077) and cfg.action_std == (0.618, 0.373)
    a = torch.rand(7, 16, 2) * 2 - 1
    assert torch.allclose(Fl.denormalise(Fl.normalise(a, cfg), cfg), a, atol=1e-6)
    assert torch.allclose(Fl.normalise(torch.tensor([0.618, 0.077 + 0.373]), cfg), torch.ones(2))


def test_loss():  # 10
    cfg = ModelConfig(d=64, depth=2, video_loss_weight=2.0, action_loss_weight=3.0)
    b = batch(cfg)
    d = Fl.draw(cfg, 4, torch.Generator().manual_seed(1))
    fake = Fake(cfg)
    total, parts = Fl.loss(fake, cfg, b, d)
    a_n = Fl.normalise(b["actions"], cfg)
    _, tgt_a = Fl.noised(a_n, d.eps_a, d.t)
    _, tgt_z = Fl.noised(b["fut"], d.eps_z, d.t)
    lv, la = ((0.3 - tgt_z) ** 2).mean(), ((-0.2 - tgt_a) ** 2).mean()
    assert torch.allclose(total, 2 * lv + 3 * la) and torch.allclose(parts["loss_video"], lv)
    assert torch.equal(fake.calls[0]["instr"], torch.where(d.uncond, NULL, b["instr"]))

    nv = ModelConfig(d=64, depth=2, video_stream=False, action_loss_weight=3.0)
    d = Fl.draw(nv, 4, torch.Generator().manual_seed(1))
    assert d.eps_z is None
    total, parts = Fl.loss(Fake(nv), nv, batch(nv), d)
    assert parts["loss_video"] is None and torch.allclose(total, 3 * parts["loss_action"])

    reg = ModelConfig(d=64, depth=2, video_stream=False, objective="regression")
    fake = Fake(reg)
    b = batch(reg)
    total, _ = Fl.loss(fake, reg, b, Fl.draw(reg, 4, torch.Generator().manual_seed(1)))
    assert torch.allclose(total, ((-0.2 - Fl.normalise(b["actions"], reg)) ** 2).mean())
    assert torch.equal(fake.calls[0]["x_a"], torch.zeros(4, 16, 2)) and torch.equal(fake.calls[0]["t"], torch.ones(4))

    p = Fl.draw(ModelConfig(), 10_000, torch.Generator().manual_seed(2)).uncond.float().mean()
    assert abs(float(p) - 0.1) < 0.01
    assert not Fl.draw(ModelConfig(), 1000, torch.Generator().manual_seed(2), p_uncond=0.0).uncond.any()


def test_euler():  # 11
    cfg = ModelConfig(d=64, depth=2)
    fake = Fake(cfg, vz=0.25, va=-0.5)
    x_z0, x_a0 = Fl.initial_noise(cfg, 3, torch.Generator().manual_seed(4))
    x_z, x_a = Fl.euler(fake, torch.zeros(3, 16, 8, 8), torch.zeros(3, 5), torch.zeros(3, dtype=torch.long),
                        x_z0, x_a0, n_steps=7, t_shift=cfg.t_shift)
    assert len(fake.calls) == 7
    assert torch.allclose(x_a, x_a0 + 0.5, atol=1e-6) and torch.allclose(x_z, x_z0 - 0.25, atol=1e-6)
    s = Fl.schedule(10, 1.65)
    assert len(s) == 11 and s[0] == 1 and s[-1] == 0
    assert abs(float(s[5]) - Fl.timeshift(1.65, 0.5)) < 1e-12
    assert torch.all(s[1:] < s[:-1])
    ts = [float(c["t"][0]) for c in fake.calls]
    assert abs(ts[0] - 1.0) < 1e-6 and ts == sorted(ts, reverse=True)
    # flow.sample draws the same initial noise from the same seed
    _, a = Fl.sample(Fake(cfg, va=0.0), cfg, torch.zeros(3, 16, 8, 8), torch.zeros(3, 5),
                     torch.zeros(3, dtype=torch.long), n_steps=3, generator=torch.Generator().manual_seed(4))
    assert torch.equal(a, x_a0)


def test_cfg():  # 12
    cfg = ModelConfig(d=64, depth=2)
    fake = Fake(cfg, by_instr=True)
    assert Fl.guided(fake, 1.0, 1.0) is fake
    obs, st, ins = torch.zeros(3, 16, 8, 8), torch.zeros(3, 5), torch.tensor([0, 1, 2])
    x_z, x_a = Fl.initial_noise(cfg, 3, torch.Generator().manual_seed(0))
    Fl.euler(fake, obs, st, ins, x_z, x_a, n_steps=2, t_shift=1.65)
    assert [c["B"] for c in fake.calls] == [3, 3]
    fake.calls.clear()
    vz, va = Fl.guided(fake, 2.0, 1.5)(obs, st, ins, x_z, x_a, torch.ones(3))
    assert fake.calls[0]["B"] == 6 and fake.calls[0]["instr"].tolist() == [0, 1, 2, NULL, NULL, NULL]
    assert torch.allclose(va, torch.full_like(va, 1 + 2.0 * (3 - 1)))
    assert torch.allclose(vz, torch.full_like(vz, -1 + 1.5 * (2 - (-1))))
    reg = ModelConfig(d=64, depth=2, video_stream=False, objective="regression")
    with pytest.raises(ValueError, match="guidance"):
        Fl.sample(Fake(reg), reg, obs, st, ins, cfg_action=2.0)
