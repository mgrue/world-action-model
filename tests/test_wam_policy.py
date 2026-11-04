"""Steps 5–8 Policy API and Checkpoints (SPEC §12 tests 13–14, 23)."""

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toycar.dataset import open_dataset  # noqa: E402
from wam import compressor as C  # noqa: E402
from wam import flow as Fl  # noqa: E402
from wam import model as M  # noqa: E402
from wam import policy as P  # noqa: E402
from wam import train as T  # noqa: E402

TINY = {"d": 64, "depth": 2}
SMOKE = {"batch": 4, "log_every": 5, "val_every": 10, "resume_every": 10, "val_windows": 16, "eval_windows": 16,
         "periodic_every": 10**9}
REPO = Path(__file__).resolve().parents[1]


def policy(tiny_root, zero=False, **kw):
    cfg = M.ModelConfig(**TINY, **kw)
    torch.manual_seed(0)
    m = M.WAM(cfg)
    if not zero:
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0, 0.1)
    return P.Policy(m, C.load("vae-v1", "cpu", root=tiny_root))


def frames(B=None):
    rng = np.random.default_rng(0)
    shape = (64, 64, 3) if B is None else (B, 64, 64, 3)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def test_act(tiny_root):  # 13
    pol = policy(tiny_root)
    st = np.linspace(-1, 1, 5).astype(np.float32)
    r = pol.act(frames(), st, "outer", n_samples=3, seed=7)
    assert r.actions.shape == (1, 3, 16, 2) and r.actions.dtype == np.float32
    assert r.actions.min() >= -1 and r.actions.max() <= 1 and r.latents is None
    assert list(r.future_k) == list(range(2, 17, 2))
    assert np.array_equal(r.actions, pol.act(frames(), st, "outer", n_samples=3, seed=7).actions)
    assert not np.array_equal(r.actions, pol.act(frames(), st, "outer", n_samples=3, seed=8).actions)
    assert np.array_equal(pol.act(frames(), st, "inner", seed=1).actions, pol.act(frames(), st, 1, seed=1).actions)
    rb = pol.act(frames(4), np.tile(st, (4, 1)), ["outer", "inner", "any", 2], n_samples=2, seed=3,
                 return_latents=True)
    assert rb.actions.shape == (4, 2, 16, 2) and rb.latents.shape == (4, 2, 8, 16, 8, 8)
    assert pol.compressor.decode(rb.latents[0]).shape == (2, 8, 64, 64, 3)
    with pytest.raises(ValueError):
        pol.act(frames(), st, "left")
    with pytest.raises(ValueError):
        pol.act(frames(4), st[None].repeat(3, 0), "any")

    zero = policy(tiny_root, zero=True)
    r = zero.act(frames(2), np.tile(st, (2, 1)), "any", n_samples=3, seed=5, n_steps=4)
    _, x_a = Fl.initial_noise(zero.config, 6, torch.Generator().manual_seed(5))
    want = Fl.denormalise(x_a, zero.config).clamp(-1, 1).reshape(2, 3, 16, 2).numpy()
    assert np.array_equal(r.actions, want)

    reg = policy(tiny_root, video_stream=False, objective="regression")
    r = reg.act(frames(), st, "any", n_samples=4, seed=0, return_latents=True)
    assert r.actions.shape == (1, 4, 16, 2) and r.latents is None and len(r.future_k) == 0
    assert all(np.array_equal(r.actions[0, 0], r.actions[0, i]) for i in range(4))
    with pytest.raises(ValueError):
        reg.act(frames(), st, "any", cfg_action=2.0)


@pytest.fixture(scope="module")
def run(tiny_data, tiny_root, tmp_path_factory):
    cfg = {**SMOKE, "steps": 4, "seed": 0, "data": str(tiny_data), "model": TINY, "name": "smoke"}
    return T.train(cfg, tmp_path_factory.mktemp("runs"), device="cpu", root=tiny_root)


def _abs_paths(obj):
    if isinstance(obj, dict):
        return [p for v in obj.values() for p in _abs_paths(v)]
    if isinstance(obj, list):
        return [p for v in obj for p in _abs_paths(v)]
    return [obj] if isinstance(obj, str) and (obj.startswith("/") or ":\\" in obj) else []


def test_run_directory(run):
    assert run.name.endswith("-smoke")
    for f in ("config.json", "log.csv", "metrics.json", "checkpoint.pt", "stdout.txt"):
        assert (run / f).exists(), f
    assert not (run / "resume.pt").exists()
    m = json.loads((run / "metrics.json").read_text())
    assert m["verdict"] == "FAIL" and m["step7"]["verdict"] == "FAIL"
    assert set(m["gates"]) == {"rmse_steer", "rmse_throttle", "commit", "follow", "any_balance", "any_two_sided"}
    header = (run / "log.csv").read_text().splitlines()[0].split(",")
    assert {"loss_video", "loss_action", "val_loss_video", "ema_val_loss_action"} <= set(header)
    assert _abs_paths(json.loads((run / "config.json").read_text())) == []


def test_promote_and_load(run, tiny_root, tmp_path):  # 14
    root = tmp_path / "ck"
    root.mkdir()
    shutil.copytree(tiny_root / "vae-v1", root / "vae-v1")
    with pytest.raises(ValueError, match="not PASS"):
        P.promote(run, "t1", root=root)
    unfinished = tmp_path / "unfinished"
    unfinished.mkdir()
    (unfinished / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        P.promote(unfinished, "t1", force=True, root=root)
    dest = P.promote(run, "t1", force=True, root=root)
    assert sorted(p.name for p in dest.iterdir()) == ["card.json", "model.pt"]
    card = json.loads((dest / "card.json").read_text())
    for k in ("name", "created", "config", "compressor", "compressor_card_sha256", "manifest_sha256", "git_commit",
              "git_dirty", "code_sha256", "metrics", "thresholds", "verdict", "eval_data_sha256", "resumes"):
        assert k in card, k
    assert card["forced"] and card["compressor"] == "vae-v1" and "model.py" in card["code_sha256"]
    assert _abs_paths(card) == []
    with pytest.raises(FileExistsError):
        P.promote(run, "t1", force=True, root=root)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pol = P.load("t1", "cpu", root=root)
    assert pol.name == "t1" and pol.card["name"] == "t1" and pol.config == M.ModelConfig(**TINY)
    ref = P.load(run, "cpu", root=root)
    assert ref.card is None and ref.run_config is not None
    x, st = frames(), np.zeros(5, np.float32)
    assert np.array_equal(pol.act(x, st, "any", seed=2).actions, ref.act(x, st, "any", seed=2).actions)

    card["code_sha256"]["model.py"] = "0" * 64
    (dest / "card.json").write_text(json.dumps(card))
    with pytest.warns(UserWarning, match="model.py"):
        P.load("t1", "cpu", root=root)
    card["code_sha256"]["model.py"] = T.code_sha256()["model.py"]
    card["code_sha256"]["evaluate.py"] = "0" * 64  # not a watched file: no warning
    (dest / "card.json").write_text(json.dumps(card))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        P.load("t1", "cpu", root=root)

    (root / "vae-v1" / "card.json").write_text(json.dumps({"name": "vae-v1", "code_sha256": C.code_sha256(),
                                                           "changed": True}))
    with pytest.raises(ValueError, match="compressor"):
        P.load("t1", "cpu", root=root)
    with pytest.raises(FileNotFoundError):
        P.load("nope", "cpu", root=root)


@pytest.mark.skipif(not (REPO / "checkpoints/wam-v1").exists() or not (REPO / "data/step3-v1").exists(),
                    reason="needs checkpoints/wam-v1 and data/step3-v1")
def test_real_checkpoint():  # 23
    pol = P.load("wam-v1", root=REPO / "checkpoints")
    es = open_dataset(REPO / "data/step3-v1", check=False).split("val")
    w = es.eval_subset(8, future_frames=())
    items = [w[i] for i in range(len(w))]
    r = pol.act(np.stack([it["obs_frame"] for it in items]), np.stack([it["state"] for it in items]),
                [int(it["instruction"]) for it in items], n_samples=2, seed=0, return_latents=True)
    assert r.actions.shape == (8, 2, 16, 2) and r.actions.min() >= -1 and r.actions.max() <= 1
    assert r.latents.shape == (8, 2, 8, 16, 8, 8)
