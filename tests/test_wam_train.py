"""Steps 5–8 training loop (SPEC §12 tests 15–18)."""

import json

import pytest

torch = pytest.importorskip("torch")

from toycar.dataset import open_dataset  # noqa: E402
from wam import compressor as C  # noqa: E402
from wam import model as M  # noqa: E402
from wam import train as T  # noqa: E402

TINY = {"d": 64, "depth": 2}
SMOKE = {"batch": 4, "log_every": 5, "val_every": 10, "resume_every": 10, "val_windows": 16, "eval_windows": 16,
         "periodic_every": 10**9}


def _state(run):
    return torch.load(run / "resume.pt", weights_only=True)


def _equal_dicts(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], dict):
            _equal_dicts(a[k], b[k])
        elif isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def test_seed_determinism(tiny_data, tiny_root, tmp_path):  # 15
    def weights(name, seed):
        cfg = {**SMOKE, "steps": 3, "seed": seed, "data": str(tiny_data), "model": TINY}
        return _state(T.train(cfg, tmp_path / name, device="cpu", root=tiny_root, _stop_at=3))["model"]

    a, b, c = weights("a", 0), weights("b", 0), weights("c", 1)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not all(torch.equal(a[k], c[k]) for k in a)


def test_resume_bit_identical(tiny_data, tiny_root, tmp_path):  # 16
    cfg = {**SMOKE, "steps": 20, "seed": 0, "data": str(tiny_data), "model": TINY}
    straight = T.train(cfg, tmp_path / "s", device="cpu", root=tiny_root, _stop_at=20)
    part = T.train(cfg, tmp_path / "r", device="cpu", root=tiny_root, _stop_at=10)
    assert (part / "resume.pt").exists() and not (part / "checkpoint.pt").exists()
    assert T.train({}, part, resume=True, device="cpu", root=tiny_root, _stop_at=20) == part
    a, b = _state(straight), _state(part)
    for k in ("model", "ema", "opt", "sched"):
        _equal_dicts(a[k], b[k])
    assert a["sampler"] == b["sampler"] and torch.equal(a["gen"], b["gen"]) and torch.equal(a["acc"], b["acc"])
    strip = lambda p: [r.split(",")[:8] + r.split(",")[10:] for r in (p / "log.csv").read_text().splitlines()]  # noqa: E731
    assert strip(straight) == strip(part)
    assert [int(r.split(",")[0]) for r in (part / "log.csv").read_text().splitlines()[1:]] == [0, 5, 10, 15, 20]

    done = T.train({}, part, resume=True, device="cpu", root=tiny_root)  # finishes: checkpoint, metrics, no resume.pt
    assert not (done / "resume.pt").exists() and (done / "checkpoint.pt").exists() and (done / "metrics.json").exists()
    ck = torch.load(done / "checkpoint.pt", weights_only=True)
    assert all(torch.equal(ck["state_dict"][k], a["ema"][k]) for k in ck["state_dict"])
    assert [r["step"] for r in json.loads((done / "config.json").read_text())["resumes"]] == [10, 20]


def test_ema_and_weight_decay():  # 17
    assert T.ema_decay(0) == pytest.approx(0.1) and T.ema_decay(5) == pytest.approx(6 / 15)
    assert T.ema_decay(10**6) == 0.999
    lin, ema = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    e0 = [p.detach().clone() for p in ema.parameters()]
    T.ema_update(ema, lin, 0.25)
    for e, p, x in zip(ema.parameters(), lin.parameters(), e0):
        assert torch.allclose(e, 0.25 * x + 0.75 * p)

    m = M.WAM(M.ModelConfig(**TINY))
    dec, nodec = T.param_groups(m, 0.01)
    assert dec["weight_decay"] == 0.01 and nodec["weight_decay"] == 0.0
    lin_w = {id(x.weight) for x in m.modules() if isinstance(x, torch.nn.Linear)}
    assert {id(p) for p in dec["params"]} == lin_w
    ids = {id(p) for p in nodec["params"]}
    assert id(m.instr.weight) in ids and id(m.frame_in.bias) in ids and id(m.blocks[0].qn.weight) in ids
    assert len(dec["params"]) + len(nodec["params"]) == len(list(m.parameters()))


def test_val_deterministic(tiny_data, tiny_root):  # 18
    comp = C.load("vae-v1", "cpu", root=tiny_root)
    ds = open_dataset(tiny_data, check=False)
    cfg = M.ModelConfig(**TINY)
    torch.manual_seed(0)
    m = M.WAM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    v1 = T.ValSet(ds, cfg, comp, 24, torch.device("cpu"))
    v2 = T.ValSet(ds, cfg, comp, 24, torch.device("cpu"))
    a, b, c = v1.losses(m), v1.losses(m), v2.losses(m)
    assert a == b == c
    assert all(v == v for v in a.values())  # no NaN
