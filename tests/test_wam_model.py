"""Steps 5–8 network (SPEC §12 tests 1–7)."""

import pytest

torch = pytest.importorskip("torch")

from wam import model as M  # noqa: E402

TINY = {"d": 64, "depth": 2}


def tiny(**kw):
    return M.ModelConfig(**TINY, **kw)


def randomised(cfg, seed=0):
    torch.manual_seed(seed)
    m = M.WAM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.1)
    return m.eval()


def inputs(cfg, B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    fut = torch.randn(B, cfg.n_future, 16, 8, 8, generator=g) if cfg.video_stream else None
    return (torch.randn(B, 16, 8, 8, generator=g), torch.randn(B, 5, generator=g), torch.tensor([0, 3])[:B],
            fut, torch.randn(B, cfg.horizon, 2, generator=g), torch.rand(B, generator=g))


def test_position_ids():  # 1
    ids = M.position_ids(M.ModelConfig())
    assert ids.shape == (162, 3) and ids.dtype == torch.long
    rows = [(0, i // 4, i % 4) for i in range(16)] + [(0, 0, 0), (0, 0, 0)]
    rows += [(k, i // 4, i % 4) for k in range(2, 17, 2) for i in range(16)]
    rows += [(k, 0, 0) for k in range(1, 17)]
    assert ids.tolist() == [list(r) for r in rows]
    counts = {"stride1": M.ModelConfig(future_stride=1), "h8": M.ModelConfig(horizon=8),
              "h4": M.ModelConfig(horizon=4), "novideo": M.ModelConfig(video_stream=False)}
    assert {k: len(M.position_ids(c)) for k, c in counts.items()} == {"stride1": 290, "h8": 90, "h4": 54, "novideo": 34}
    assert {k: c.n_tokens for k, c in counts.items()} == {"stride1": 290, "h8": 90, "h4": 54, "novideo": 34}
    assert M.ModelConfig().future_frames == (1, 3, 5, 7, 9, 11, 13, 15)
    assert M.ModelConfig(video_stream=False).future_frames == ()


def test_patchify():  # 2
    z = torch.randn(3, 16, 8, 8)
    tok = M.patchify(z)
    assert tok.shape == (3, 16, 64)
    assert torch.equal(M.unpatchify(tok), z)
    for i in (0, 5, 15):
        h, w = i // 4, i % 4
        cell = z[:, :, 2 * h : 2 * h + 2, 2 * w : 2 * w + 2]  # (B, c, ph, pw) → channels (c ph pw)
        assert torch.equal(tok[:, i], cell.reshape(3, 64))


def test_param_count():  # 3
    assert M.n_params(M.WAM(M.ModelConfig())) == 8_229_888


@pytest.mark.parametrize("kw", [{}, {"video_stream": False}, {"horizon": 8}, {"future_stride": 1}])
def test_zero_init_heads(kw):  # 4
    cfg = tiny(**kw)
    m = M.WAM(cfg)
    v_z, v_a = m(*inputs(cfg))
    assert v_a.shape == (2, cfg.horizon, 2) and torch.equal(v_a, torch.zeros_like(v_a))
    if cfg.video_stream:
        assert v_z.shape == (2, cfg.n_future, 16, 8, 8) and torch.equal(v_z, torch.zeros_like(v_z))
    else:
        assert v_z is None


def test_config_validation():  # 5
    bad = [{"objective": "regression"}, {"video_loss_weight": 0.0}, {"horizon": 12}, {"future_stride": 3},
           {"d": 96}, {"objective": "diffusion"}, {"p_uncond": 1.0}]
    for kw in bad:
        with pytest.raises(ValueError):
            M.ModelConfig(**kw)
    M.ModelConfig(objective="regression", video_stream=False)
    M.ModelConfig(video_stream=False, video_loss_weight=0.0)
    with pytest.raises(ValueError, match="unknown"):
        M.ModelConfig.from_dict({"nope": 1})
    cfg = M.ModelConfig(horizon=8)
    assert M.ModelConfig.from_dict(cfg.to_dict()) == cfg


def test_rope_relative():  # 6
    cfg = tiny()
    m = randomised(cfg)
    with torch.no_grad():
        x, vec, mod = m.embed(*inputs(cfg))
        a = m.trunk(x, mod)
        ids = M.position_ids(cfg)
        ids[:, 0] += 7
        b = m.trunk(x, mod, M.rope_freqs(ids))
        ids[:, 1] += 3  # a common shift on any axis
        c = m.trunk(x, mod, M.rope_freqs(ids))
    assert not torch.equal(a, torch.zeros_like(a))
    assert torch.allclose(a, b, atol=1e-4) and torch.allclose(a, c, atol=1e-4)


def test_token_permutation():  # 7
    cfg = tiny()
    m = randomised(cfg)
    perm = torch.randperm(cfg.n_tokens, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        x, vec, mod = m.embed(*inputs(cfg))
        out = m.trunk(x, mod)
        out_p = m.trunk(x[:, perm], tuple(t[:, perm] for t in mod), m.pe[perm])
    assert torch.allclose(out[:, perm], out_p, atol=1e-5)
