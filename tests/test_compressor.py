"""Step 4 frame compressor (SPEC §10 tests 1–12, plus the real-Checkpoint test)."""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toycar.dataset import open_dataset  # noqa: E402
from wam import compressor as C  # noqa: E402

TINY = {"channels": [8, 8, 16, 16], "blocks": 1, "z": 16}
SMOKE = {"arch": TINY, "batch": 4, "stats_frames": 64, "eval_frames": 32, "val_frames": 16,
         "log_every": 5, "val_every": 10, "resume_every": 10}
REPO = Path(__file__).resolve().parents[1]


def tiny_compressor(seed=0):
    torch.manual_seed(seed)
    vae = C.VAE(TINY)
    mean = torch.linspace(-0.5, 0.5, 16)
    std = torch.linspace(0.5, 2.0, 16)
    return C.Compressor(vae, mean, std, device="cpu")


@pytest.fixture(scope="module")
def frames(tiny_data):
    es = open_dataset(tiny_data, check=False).split("val")
    return es.frames(0, ticks=range(17))  # (17, 64, 64, 3)


def test_toycar_has_no_torch():  # 1
    mods = sorted(p.stem for p in (REPO / "src/toycar").glob("*.py") if p.stem != "__main__")
    code = ("import sys\n" + "".join(f"import toycar.{m}\n" for m in mods if m != "__init__")
            + "import toycar\nprint('torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_network_shapes_and_params():  # 2
    vae = C.VAE(TINY)
    mu, logvar = vae.encode(torch.zeros(3, 3, 64, 64))
    assert vae.enc(torch.zeros(3, 3, 64, 64)).shape == (3, 32, 8, 8)
    assert mu.shape == logvar.shape == (3, 16, 8, 8)
    assert vae.decode(mu).shape == (3, 3, 64, 64)
    assert C.n_params(C.VAE()) == 3_166_435
    with pytest.raises(ValueError):
        C.VAE({"channels": [8, 8, 16, 16], "blocks": 1, "z": 8})


def test_encode_decode_shapes(frames):  # 3
    comp = tiny_compressor()
    z = comp.encode(frames)
    assert z.dtype == torch.float32 and z.shape == (17, 16, 8, 8) and z.device == comp.device
    assert torch.equal(comp.encode(torch.from_numpy(frames)), z)
    stacked = np.stack([frames, frames])
    z2 = comp.encode(stacked)
    assert z2.shape == (2, 17, 16, 8, 8)
    assert comp.encode(frames[0]).shape == (16, 8, 8)
    x = comp.decode(z2)
    assert x.dtype == torch.uint8 and x.shape == (2, 17, 64, 64, 3)
    xf = comp.decode(z2, as_float=True)
    assert xf.dtype == torch.float32 and xf.shape == (2, 17, 3, 64, 64)
    assert xf.min() >= -1 and xf.max() <= 1
    with pytest.raises(ValueError):
        comp.encode(frames.astype(np.float32))
    with pytest.raises(ValueError):
        comp.decode(torch.zeros(2, 8, 8, 16))


def test_normalisation(frames):  # 4
    comp = tiny_compressor()
    x = C.to_x(torch.from_numpy(frames))
    with torch.no_grad():
        mu, _ = comp.vae.encode(x)
        raw = comp.vae.decode(mu)
    z = comp.encode(frames)
    m, s = comp.latent_mean.view(1, 16, 1, 1), comp.latent_std.view(1, 16, 1, 1)
    assert torch.allclose(z, (mu - m) / s, atol=1e-6)
    assert torch.allclose(comp.decode(z, as_float=True), raw.clamp(-1, 1), atol=1e-4)
    assert torch.equal(comp.decode(z), C.to_u8(comp.decode(z, as_float=True)))


def test_encode_exactness(frames):  # 5
    comp = tiny_compressor()
    a, b = comp.encode(frames), comp.encode(frames)
    assert torch.equal(a, b)
    one = comp.encode(frames[5:6])
    assert torch.allclose(one[0], a[5], atol=1e-5)


def test_encode_windows(tiny_data):  # 6
    es = open_dataset(tiny_data, check=False).split("val")
    w = es.windows()
    items = [w[i] for i in (0, 7, 20)]
    batch = {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in ("obs_frame", "future_frames")}
    comp = tiny_compressor()
    obs, fut = comp.encode_windows(batch)
    assert obs.shape == (3, 16, 8, 8) and fut.shape == (3, 16, 16, 8, 8)
    assert torch.allclose(obs, comp.encode(batch["obs_frame"]), atol=1e-5)
    assert torch.allclose(fut, comp.encode(batch["future_frames"]), atol=1e-5)
    e, t = int(items[1]["episode"]), int(items[1]["t"])
    for k in (0, 5, 15):  # future latent k ↔ Frame t + 1 + k
        assert torch.allclose(fut[1, k], comp.encode(es.frames(e, ticks=[t + 1 + k])[0]), atol=1e-5)


def test_token_order():  # 7
    z = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 8, 8)
    tok = z.flatten(-2).transpose(-1, -2)
    assert tok.shape == (64, 16)
    for i in (0, 1, 8, 9, 63):
        assert torch.equal(tok[i], z[:, i // 8, i % 8])


def test_decode_gradients():  # 8
    comp = tiny_compressor()
    z = torch.randn(2, 16, 8, 8, requires_grad=True)
    comp.decode(z, as_float=True).square().mean().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in comp.vae.parameters())


def _train(tiny_data, out, steps, seed=0, **kw):
    cfg = {**SMOKE, "steps": steps, "seed": seed, "data": str(tiny_data), **kw}
    return C.train(cfg, out, device="cpu")


def _weights(run):
    return torch.load(run / "checkpoint.pt", weights_only=True)["state_dict"]


def test_seed_determinism(tiny_data, tmp_path):  # 9
    a = _weights(_train(tiny_data, tmp_path / "a", 3))
    b = _weights(_train(tiny_data, tmp_path / "b", 3))
    c = _weights(_train(tiny_data, tmp_path / "c", 3, seed=1))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_resume_bit_identical(tiny_data, tmp_path):  # 10
    straight = _train(tiny_data, tmp_path / "s", 20)
    cfg = {**SMOKE, "steps": 20, "seed": 0, "data": str(tiny_data)}
    part = C.train(cfg, tmp_path / "r", device="cpu", _stop_at=10)
    assert (part / "resume.pt").exists() and not (part / "checkpoint.pt").exists()
    done = C.train({}, part, resume=True, device="cpu")
    assert done == part and not (part / "resume.pt").exists()
    a, b = _weights(straight), _weights(done)
    assert all(torch.equal(a[k], b[k]) for k in a)
    ca = torch.load(straight / "checkpoint.pt", weights_only=True)
    cb = torch.load(done / "checkpoint.pt", weights_only=True)
    assert torch.equal(ca["latent_std"], cb["latent_std"])
    cfg = json.loads((done / "config.json").read_text())
    assert [r["step"] for r in cfg["resumes"]] == [10]
    log = (done / "log.csv").read_text().splitlines()
    assert [int(r.split(",")[0]) for r in log[1:]] == [5, 10, 15, 20]
    strip = lambda rows: [r.split(",")[:5] for r in rows]  # noqa: E731  (timings differ)
    assert strip(log) == strip((straight / "log.csv").read_text().splitlines())


@pytest.fixture(scope="module")
def run(tiny_data, tmp_path_factory):
    return _train(tiny_data, tmp_path_factory.mktemp("runs"), 4, name="smoke")


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
    m = json.loads((run / "metrics.json").read_text())
    assert set(m["check"]) == {"clean", "noisy", "probes"}
    assert set(m["check"]["probes"]) == set(C.COLOUR_PROBES)
    assert len(m["latent_std"]) == 16 and m["params"] == C.n_params(C.VAE(TINY))
    assert "training" in (run / "stdout.txt").read_text()


def test_promote_and_load(run, tmp_path):  # 11
    root = tmp_path / "ck"
    with pytest.raises(ValueError, match="not PASS"):
        C.promote(run, "t1", root=root)
    dest = C.promote(run, "t1", force=True, root=root)
    assert sorted(p.name for p in dest.iterdir()) == ["card.json", "model.pt"]
    card = json.loads((dest / "card.json").read_text())
    for k in ("name", "created", "source_run", "git_commit", "git_dirty", "code_sha256", "torch_version", "device",
              "dataset", "manifest_sha256", "split", "stats_frames", "config", "final", "check", "latent_std"):
        assert k in card, k
    assert card["name"] == "t1" and card["forced"] and card["check"]["clean"]["verdict"] != "PASS"
    assert _abs_paths(card) == []
    with pytest.raises(FileExistsError):
        C.promote(run, "t1", force=True, root=root)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comp = C.load("t1", "cpu", root=root)
    assert comp.name == "t1" and comp.card["name"] == "t1" and comp.arch == TINY
    ref = C.load(run, "cpu")
    assert ref.name is None
    x = np.zeros((2, 64, 64, 3), np.uint8)
    assert torch.equal(comp.encode(x), ref.encode(x))

    card["code_sha256"] = "0" * 64
    (dest / "card.json").write_text(json.dumps(card))
    with pytest.warns(UserWarning, match="code_sha256"):
        C.load("t1", "cpu", root=root)

    ck = torch.load(dest / "model.pt", weights_only=True)
    ck["arch"] = {**TINY, "channels": [8, 8, 16, 32]}
    torch.save(ck, dest / "model.pt")
    with pytest.raises(ValueError, match="does not fit"):
        C.load("t1", "cpu", root=root)
    with pytest.raises(FileNotFoundError):
        C.load("nope", "cpu", root=root)


def test_cli_smoke(tiny_data, tmp_path, capsys):  # 12
    runs, root = tmp_path / "runs", tmp_path / "ck"
    code = C.main(["train", "--steps", "5", "--data", str(tiny_data), "--cpu", "--runs", str(runs),
                   "--arch", json.dumps(TINY), "--batch", "4", "--eval-frames", "32", "--stats-frames", "64",
                   "--val-frames", "16", "--name", "cli"])
    assert code == 1  # 5 steps cannot pass
    (run,) = list(runs.iterdir())
    assert C.main(["eval", str(run), "--cpu", "--eval-frames", "32", "--json", str(tmp_path / "e.json")]) == 1
    assert json.loads((tmp_path / "e.json").read_text())["clean"]["all"]["verdict"] in ("FAIL", "INVALID")
    assert C.main(["promote", str(run), "--name", "x", "--root", str(root)]) == 1
    assert C.main(["promote", str(run), "--name", "x", "--force", "--root", str(root)]) == 0
    assert C.main(["eval", "x", "--root", str(root), "--data", str(tiny_data), "--cpu", "--eval-frames", "32"]) == 1
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert C.main(["show", str(run), "--cpu", "--eval-frames", "32"]) == 2
        pytest.skip("matplotlib not installed; show exits 2")
    out = tmp_path / "c.png"
    assert C.main(["show", "x", "--root", str(root), "--data", str(tiny_data), "--cpu", "--eval-frames", "32",
                   "--out", str(out)]) == 0
    assert out.stat().st_size > 1000


@pytest.mark.skipif(not (REPO / "checkpoints/vae-v1").exists() or not (REPO / "data/step3-v1").exists(),
                    reason="needs checkpoints/vae-v1 and data/step3-v1")
def test_real_checkpoint():
    comp = C.load("vae-v1", root=REPO / "checkpoints")
    res = C.evaluate(comp, REPO / "data/step3-v1", 512, noisy=False, probes=())
    assert res["summary"]["clean"]["all"]["verdict"] == "PASS"
