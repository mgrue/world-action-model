"""The world action model as a Policy, its Checkpoints and its CLI (Steps 5–8 SPEC §8, §10, §11).

    caffeinate -i uv run python -m wam.policy train [--name NAME] [--steps 32000 | --quick] [--seed 0]
                                                    [--data data/step3-v1] [--set key=value ...] [--cpu]
    caffeinate -i uv run python -m wam.policy train --resume runs/wam/<run>
    uv run python -m wam.policy eval    <run-dir | checkpoint-name> [--json OUT] [--sweeps]
    uv run --group viz python -m wam.policy show <run-dir | checkpoint-name> [--out samples.png]
    uv run python -m wam.policy promote runs/wam/<run> --name wam-v1[-variant] [--force]

Step 9 uses `load("wam-v1").act(frame, state.features(), instruction, seed=…)`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from . import compressor as C
from . import flow
from .model import NULL, WAM, ModelConfig
from .runs import CHECKPOINTS, _now, _read_weights, _rel, data_path

INSTRUCTION_IDS = {"outer": 0, "inner": 1, "any": 2, "null": NULL}
WATCHED = ("model.py", "flow.py")  # files that define what the weights mean (SPEC §16.6)


@dataclass
class ActResult:
    actions: np.ndarray  # float32 (B, n_samples, H, 2), raw units, clipped to [−1, 1]
    latents: Tensor | None  # (B, n_samples, K, 16, 8, 8) normalised; policy.compressor.decode → Frames
    future_k: np.ndarray  # (K,)


class Policy:
    """A trained WAM (EMA weights) with its frozen compressor."""

    def __init__(self, model: WAM, compressor: C.Compressor, *, card: dict | None = None, name: str | None = None,
                 source: Path | None = None, run_config: dict | None = None) -> None:
        self.compressor = compressor
        self.device = compressor.device
        self.model = model.to(self.device).eval()
        self.config: ModelConfig = model.cfg
        self.card = card
        self.name = name
        self.source = source
        self.run_config = run_config
        self.horizon = self.config.horizon
        self.future_k = self.config.future_k

    def _instr(self, instruction, B: int) -> Tensor:
        def one(v) -> int:
            if isinstance(v, str):
                if v not in INSTRUCTION_IDS:
                    raise ValueError(f"unknown Instruction {v!r}; known: {list(INSTRUCTION_IDS)}")
                return INSTRUCTION_IDS[v]
            i = int(v)
            if not 0 <= i <= NULL:
                raise ValueError(f"Instruction id must be in 0..{NULL}, got {i}")
            return i

        if isinstance(instruction, (str, int, np.integer)):
            ids = [one(instruction)] * B
        else:
            ids = [one(v) for v in instruction]
            if len(ids) != B:
                raise ValueError(f"{len(ids)} Instructions for a batch of {B}")
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def act(self, frame, state, instruction, *, n_samples: int = 1, n_steps: int = 10, cfg_action: float = 1.0,
            cfg_video: float = 1.0, seed: int | torch.Generator | None = None,
            return_latents: bool = False) -> ActResult:
        """frame uint8 (64,64,3) or (B,64,64,3); state = State.features() (5,) or (B,5); instruction a name or id,
        or a sequence of B. Each input is repeated n_samples times along the batch."""
        fr = torch.from_numpy(np.ascontiguousarray(frame)) if isinstance(frame, np.ndarray) else frame
        if fr.dim() == 3:
            fr = fr[None]
        st = torch.as_tensor(np.asarray(state, np.float32) if not isinstance(state, Tensor) else state,
                             dtype=torch.float32)
        if st.dim() == 1:
            st = st[None]
        B = fr.shape[0]
        if tuple(st.shape) != (B, 5):
            raise ValueError(f"state must be (5,) or ({B}, 5), got {tuple(st.shape)}")
        if n_samples < 1:
            raise ValueError(f"n_samples must be ≥ 1, got {n_samples}")
        ins = self._instr(instruction, B)
        obs = self.compressor.encode(fr).clone()
        rep = lambda x: x.repeat_interleave(n_samples, 0)  # noqa: E731
        if isinstance(seed, torch.Generator):
            g = seed
        else:
            g = torch.Generator()
            g.manual_seed(int(seed)) if seed is not None else g.seed()
        if self.config.objective == "regression":  # deterministic: one pass per input, identical copies
            z, a_n = self.sample(obs, st.to(self.device), ins, n_steps=n_steps, cfg_action=cfg_action,
                                 cfg_video=cfg_video, generator=g)
            a_n = rep(a_n)
        else:
            z, a_n = self.sample(rep(obs), rep(st.to(self.device)), rep(ins), n_steps=n_steps,
                                 cfg_action=cfg_action, cfg_video=cfg_video, generator=g)
        acts = flow.denormalise(a_n, self.config).clamp(-1, 1)
        acts = acts.reshape(B, n_samples, self.horizon, 2).float().cpu().numpy()
        lat = None
        if return_latents and z is not None:
            lat = z.reshape(B, n_samples, *z.shape[1:])
        return ActResult(acts, lat, self.future_k.copy())

    @torch.no_grad()
    def sample(self, obs_z: Tensor, state: Tensor, instr: Tensor, *, n_steps: int = 10, cfg_action: float = 1.0,
               cfg_video: float = 1.0, generator: torch.Generator | None = None) -> tuple[Tensor | None, Tensor]:
        """(B,16,8,8), (B,5), (B,) long → (future Latents (B,K,16,8,8) normalised | None, actions (B,H,2) normalised)."""
        return flow.sample(self.model, self.config, obs_z.to(self.device), state.to(self.device),
                           instr.to(self.device), n_steps=n_steps, cfg_action=cfg_action, cfg_video=cfg_video,
                           generator=generator)


# -- Checkpoints (SPEC §10) -----------------------------------------------------------------
def _build(ck: dict, where: str) -> WAM:
    try:
        cfg = ModelConfig.from_dict(ck["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: config does not build: {exc}") from None
    model = WAM(cfg)
    try:
        model.load_state_dict(ck["state_dict"])
    except RuntimeError as exc:
        raise ValueError(f"{where}: state dict does not fit the config: {exc}") from None
    return model


def load(name: str | Path, device: str | torch.device | None = None, *, root: Path = CHECKPOINTS) -> Policy:
    """A Policy from `root/<name>/` (model.pt + card.json) or a Run directory (checkpoint.pt + config.json).
    EMA weights; the compressor is loaded from `root` by the name recorded with the weights."""
    from .train import compressor_card_sha256

    p, root = Path(name), Path(root)
    card = run_config = None
    if p.is_dir() and (p / "checkpoint.pt").exists():
        ck = _read_weights(p / "checkpoint.pt")
        where, pname = str(p), None
        if (p / "config.json").exists():
            run_config = json.loads((p / "config.json").read_text())
    else:
        d = root / str(name)
        if not (d / "model.pt").exists():
            raise FileNotFoundError(f"no Checkpoint {name!r} in {root} (and not a finished Run directory)")
        card = json.loads((d / "card.json").read_text())
        ck = _read_weights(d / "model.pt")
        where, pname, p = str(d), str(name), d
    model = _build(ck, where)
    comp_name = ck["compressor"]
    have = compressor_card_sha256(comp_name, root)
    if have != ck["compressor_card_sha256"]:
        raise ValueError(f"{where}: compressor {comp_name!r} in {root} is not the one these weights were trained "
                         "with (card.json sha256 mismatch)")
    if card is not None:
        from .train import code_sha256

        now, then = code_sha256(), card.get("code_sha256") or {}
        changed = [f for f in WATCHED if then.get(f) != now.get(f)]
        if changed:
            warnings.warn(f"Checkpoint {name!r}: {', '.join('src/wam/' + f for f in changed)} changed since it was "
                          "trained (code_sha256 differs); the weights still load", stacklevel=2)
    comp = C.load(comp_name, device, root=root)
    return Policy(model, comp, card=card, name=pname, source=p, run_config=run_config)


def promote(run: Path, name: str, *, force: bool = False, root: Path = CHECKPOINTS) -> Path:
    """Copy a finished Run's EMA weights to `root/<name>/` with its card. Never overwrites."""
    run, root = Path(run), Path(root)
    dest = root / name
    if dest.exists():
        raise FileExistsError(f"Checkpoint {name!r} already exists in {root}; pick a new name")
    for f in ("checkpoint.pt", "config.json", "metrics.json"):
        if not (run / f).exists():
            raise FileNotFoundError(f"{run} is not a finished Run (missing {f})")
    cfg = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    verdict = metrics["verdict"]
    if verdict != "PASS" and not force:
        raise ValueError(f"{run}: verdict is {verdict}, not PASS; pass --force to promote anyway")
    _build(_read_weights(run / "checkpoint.pt"), str(run))
    prov = cfg["provenance"]
    card = {
        "name": name, "created": _now(), "source_run": _rel(run),
        "config": cfg["model"],
        "recipe": {k: v for k, v in cfg.items() if k not in ("model", "provenance", "resumes", "created")},
        **{k: prov[k] for k in ("compressor", "compressor_card_sha256", "dataset", "manifest_sha256", "git_commit",
                                "git_dirty", "code_sha256", "torch_version", "device")},
        "metrics": {k: metrics.get(k) for k in ("step7", "gates", "general", "fork", "report")},
        "thresholds": metrics.get("thresholds"), "verdict": verdict, "forced": bool(force and verdict != "PASS"),
        "eval_data_sha256": metrics.get("eval_data_sha256"), "resumes": cfg.get("resumes", []),
        "params": metrics.get("params"), "train_s": metrics.get("train_s"),
    }
    tmp = root / f".{name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copyfile(run / "checkpoint.pt", tmp / "model.pt")
    (tmp / "card.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
    tmp.rename(dest)
    return dest


# -- CLI (SPEC §11) -------------------------------------------------------------------------
def parse_set(items: Sequence[str]) -> dict:
    """`key=value` switches → a partial ModelConfig dict (validated by the caller)."""
    from dataclasses import fields

    known = {f.name for f in fields(ModelConfig)}
    out: dict = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--set needs key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in known:
            raise ValueError(f"unknown switch {k!r}; known: {sorted(known)}")
        low = v.strip().lower()
        if low in ("true", "false"):
            val = low == "true"
        else:
            try:
                val = json.loads(v)
            except json.JSONDecodeError:
                val = v.strip()
        out[k] = val
    return out


def _target_data(policy: Policy, override: str | None) -> Path:
    if override:
        return Path(override)
    src = policy.card or policy.run_config or {}
    return data_path(src.get("dataset") or src.get("data") or "data/step3-v1")


def _device(a) -> str | None:
    return "cpu" if a.cpu else None


def cmd_train(a) -> int:
    from . import train as T

    if a.resume:
        run = T.train({}, a.resume, resume=True, device=_device(a), root=a.root)
    else:
        try:
            model = parse_set(a.set or [])
            if a.arch:
                model.update({k: v for k, v in json.loads(a.arch).items()})
            ModelConfig.from_dict(model)
        except (ValueError, TypeError) as exc:
            print(f"invalid configuration: {exc}", file=sys.stderr)
            return 2
        cfg: dict = {"name": a.name, "steps": T.QUICK_STEPS if a.quick else a.steps, "seed": a.seed,
                     "data": a.data, "model": model}
        if a.batch is not None:
            cfg["batch"] = a.batch
        if a.eval_windows is not None:
            cfg["eval_windows"] = a.eval_windows
            cfg["val_windows"] = min(T.DEFAULT_RECIPE["val_windows"], a.eval_windows)
        run = T.train(cfg, a.runs, device=_device(a), root=a.root)
    if not (run / "metrics.json").exists():
        return 1
    return 0 if json.loads((run / "metrics.json").read_text())["verdict"] == "PASS" else 1


def _open(path: Path):
    from toycar.dataset import open_dataset

    return open_dataset(path, check=False)


def cmd_eval(a) -> int:
    from . import evaluate as E
    from .train import read_log

    policy = load(a.target, _device(a), root=a.root)
    ds = _open(_target_data(policy, a.data))
    res = E.evaluate(policy, ds, sz=E.sizes(a.eval_windows), sweeps=a.sweeps, quiet=True)
    if policy.card is None and (Path(a.target) / "log.csv").exists():
        s7 = E.step7(read_log(Path(a.target)), policy.config.video_stream)
    else:
        s7 = ((policy.card or {}).get("metrics") or {}).get("step7") or {"verdict": "FAIL", "streams": {}}
    E.print_report(res, s7)
    verdict = "PASS" if s7["verdict"] == "PASS" and res["verdict"] == "PASS" else "FAIL"
    if a.json:
        out = {"verdict": verdict, "step7": s7, **{k: v for k, v in res.items() if k != "verdict"},
               "gates_verdict": res["verdict"]}
        Path(a.json).write_text(json.dumps(E.jsonable(out), indent=2, allow_nan=False) + "\n")
    print(f"verdict: {verdict}")
    return 0 if verdict == "PASS" else 1


def cmd_show(a) -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("show needs matplotlib: uv sync --group viz (or uv run --group viz …)", file=sys.stderr)
        return 2
    from . import evaluate as E

    policy = load(a.target, _device(a), root=a.root)
    ds = _open(_target_data(policy, a.data))
    sz = E.sizes(a.eval_windows)
    sub, ref = E.subsets(ds, policy.horizon, sz), E.fork_references(ds, policy.horizon)
    is_run = policy.card is None
    out = a.out or (Path(a.target) / "samples.png" if is_run else Path("samples.png"))
    E.contact_sheet(out, policy.name or Path(a.target).name, E.sheet_data(policy, ds, sub, ref, sz))
    print(out)
    return 0


def cmd_promote(a) -> int:
    try:
        dest = promote(a.run, a.name, force=a.force, root=a.root)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(dest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from .train import DEFAULT_DATA, DEFAULT_RECIPE, RUNS_DIR

    ap = argparse.ArgumentParser(prog="python -m wam.policy", description="Train, check and freeze the world action model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    hide = argparse.SUPPRESS

    tr = sub.add_parser("train", help="train a WAM into a new Run directory (or --resume one)")
    tr.add_argument("--name", default="wam")
    tr.add_argument("--steps", type=int, default=DEFAULT_RECIPE["steps"])
    tr.add_argument("--quick", action="store_true", help="3,200 steps")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--data", default=DEFAULT_DATA)
    tr.add_argument("--set", action="append", metavar="KEY=VALUE", help="a ModelConfig switch, e.g. video_stream=false")
    tr.add_argument("--cpu", action="store_true")
    tr.add_argument("--resume", type=Path, metavar="RUN")
    tr.add_argument("--runs", type=Path, default=RUNS_DIR, help=hide)
    tr.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)
    tr.add_argument("--arch", help=hide)
    tr.add_argument("--batch", type=int, help=hide)
    tr.add_argument("--eval-windows", type=int, help=hide)

    for cmd, hlp in (("eval", "print the gate table and report; exit 0 on PASS"),
                     ("show", "write the contact sheet (needs --group viz)")):
        p = sub.add_parser(cmd, help=hlp)
        p.add_argument("target", help="Run directory or Checkpoint name")
        p.add_argument("--data", help="dataset directory (default: the one the model was trained on)")
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)
        p.add_argument("--eval-windows", type=int, help=hide)
        if cmd == "eval":
            p.add_argument("--json", type=Path, metavar="OUT")
            p.add_argument("--sweeps", action="store_true", help="add the CFG and n_steps sweeps (≈ 25 min)")
        else:
            p.add_argument("--out", type=Path)

    pr = sub.add_parser("promote", help="freeze a Run as checkpoints/<name>/")
    pr.add_argument("run", type=Path)
    pr.add_argument("--name", required=True)
    pr.add_argument("--force", action="store_true")
    pr.add_argument("--root", type=Path, default=CHECKPOINTS, help=hide)

    a = ap.parse_args(argv)
    return {"train": cmd_train, "eval": cmd_eval, "show": cmd_show, "promote": cmd_promote}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
