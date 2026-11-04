"""The world action model network (Steps 5–6 SPEC §3–§5.5, `.scratch/step5-8-model/SPEC.md`).

`ModelConfig` holds the architecture, every training switch and the action-normalisation constants.
`WAM` is a small FLUX-style transformer over one fixed token sequence per Window:
observed Frame (16) · State (1) · Instruction (1) · future Frames (K × 16) · actions (H).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

HEAD = 64  # head size, fixed by the RoPE split
ROPE_AXES = (32, 16, 16)  # t, h, w per head
ROPE_THETA = 10_000.0
T_EMBED = 256  # sinusoid width
PATCH = 2
GRID = 8 // PATCH  # 4 × 4 tokens per Latent
FRAME_TOKENS = GRID * GRID  # 16
PATCH_DIM = 16 * PATCH * PATCH  # 64
STATE_DIM = 5
ACTION_DIM = 2
NULL = 3  # the Null Instruction's id (outer 0, inner 1, any 2 come from Step 3)
N_INSTR = 4
STREAMS = ("obs", "state", "instr", "fut", "act")
OBJECTIVES = ("flow", "regression")
HORIZONS = (4, 8, 16)
STRIDES = (1, 2)


# -- configuration (SPEC §4.3, §5.1, §5.5) --------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    d: int = 256
    depth: int = 8
    mlp_ratio: float = 3.0
    objective: str = "flow"
    video_stream: bool = True
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    t_shift: float = 1.65
    horizon: int = 16
    future_stride: int = 2
    p_uncond: float = 0.1
    action_mean: tuple[float, float] = (0.0, 0.077)
    action_std: tuple[float, float] = (0.618, 0.373)

    def __post_init__(self) -> None:
        def fix(name, typ):
            v = getattr(self, name)
            if typ is bool:
                if not isinstance(v, (bool, np.bool_)):
                    raise ValueError(f"{name} must be a bool, got {v!r}")
            elif typ is int:
                ok = isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
                if isinstance(v, bool) or not ok:
                    raise ValueError(f"{name} must be an integer, got {v!r}")
            elif typ is float:
                if isinstance(v, bool) or not isinstance(v, (int, float, np.number)) or not math.isfinite(v):
                    raise ValueError(f"{name} must be a finite number, got {v!r}")
            object.__setattr__(self, name, typ(v))

        for f in ("d", "depth", "horizon", "future_stride"):
            fix(f, int)
        for f in ("mlp_ratio", "video_loss_weight", "action_loss_weight", "t_shift", "p_uncond"):
            fix(f, float)
        fix("video_stream", bool)
        for f in ("action_mean", "action_std"):
            v = tuple(float(x) for x in getattr(self, f))
            if len(v) != ACTION_DIM:
                raise ValueError(f"{f} must have {ACTION_DIM} values, got {v}")
            object.__setattr__(self, f, v)

        if self.d <= 0 or self.d % HEAD:
            raise ValueError(f"d must be a positive multiple of {HEAD} (the head size), got {self.d}")
        if self.depth < 1:
            raise ValueError(f"depth must be ≥ 1, got {self.depth}")
        if self.mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {self.mlp_ratio}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}, got {self.horizon}")
        if self.future_stride not in STRIDES:
            raise ValueError(f"future_stride must be one of {STRIDES}, got {self.future_stride}")
        if not 0.0 <= self.p_uncond < 1.0:
            raise ValueError(f"p_uncond must be in [0, 1), got {self.p_uncond}")
        if self.t_shift <= 0:
            raise ValueError(f"t_shift must be > 0, got {self.t_shift}")
        if self.video_loss_weight < 0 or self.action_loss_weight <= 0:
            raise ValueError("video_loss_weight must be ≥ 0 and action_loss_weight > 0")
        if any(s <= 0 for s in self.action_std):
            raise ValueError(f"action_std must be > 0, got {self.action_std}")
        if self.objective == "regression" and self.video_stream:
            raise ValueError("objective=regression requires video_stream=false")
        if self.video_stream and self.video_loss_weight == 0:
            raise ValueError("video_loss_weight=0 is only allowed with video_stream=false "
                             "(low-t future Frames would leak the actions' outcome)")

    # derived layout
    @property
    def future_k(self) -> np.ndarray:
        """Ticks k of the future Frames, e.g. [2, 4, …, 16]; empty without video."""
        if not self.video_stream:
            return np.zeros(0, np.int64)
        return np.arange(self.future_stride, self.horizon + 1, self.future_stride, dtype=np.int64)

    @property
    def future_frames(self) -> tuple[int, ...]:
        """Step 3's 0-based `future_frames` argument (future Frame k ↔ index k − 1)."""
        return tuple(int(k) - 1 for k in self.future_k)

    @property
    def n_future(self) -> int:
        return len(self.future_k)

    @property
    def stream_lengths(self) -> dict[str, int]:
        lens = {"obs": FRAME_TOKENS, "state": 1, "instr": 1}
        if self.video_stream:
            lens["fut"] = FRAME_TOKENS * self.n_future
        lens["act"] = self.horizon
        return lens

    @property
    def n_tokens(self) -> int:
        return sum(self.stream_lengths.values())

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["action_mean"], d["action_std"] = list(self.action_mean), list(self.action_std)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown ModelConfig keys: {unknown}; known: {sorted(known)}")
        return cls(**d)

    def replace(self, **kw) -> ModelConfig:
        return ModelConfig.from_dict({**self.to_dict(), **kw})


# -- positions and patches (SPEC §3.1, §3.2, §3.4) ------------------------------------------
def _grid_ids() -> Tensor:
    return torch.cartesian_prod(torch.arange(GRID), torch.arange(GRID))  # 16 × (h, w), row-major


def position_ids(cfg: ModelConfig) -> Tensor:
    """(N, 3) long (t, h, w) per token, in the order obs, State, Instruction, future Frames, actions."""
    grid = _grid_ids()
    zeros = lambda n: torch.zeros(n, 1, dtype=torch.long)  # noqa: E731
    rows = [torch.cat([zeros(FRAME_TOKENS), grid], 1), torch.zeros(2, 3, dtype=torch.long)]
    for k in cfg.future_k:
        rows.append(torch.cat([torch.full((FRAME_TOKENS, 1), int(k)), grid], 1))
    H = cfg.horizon
    rows.append(torch.stack([torch.arange(1, H + 1), torch.zeros(H, dtype=torch.long),
                             torch.zeros(H, dtype=torch.long)], 1))
    return torch.cat(rows)


def patchify(z: Tensor) -> Tensor:
    """(…, 16, 8, 8) Latents → (…, 16 tokens, 64); token i = cell (i // 4, i % 4), channels (c ph pw)."""
    return rearrange(z, "... c (h ph) (w pw) -> ... (h w) (c ph pw)", ph=PATCH, pw=PATCH)


def unpatchify(tok: Tensor) -> Tensor:
    """Inverse of `patchify`: (…, 16, 64) → (…, 16, 8, 8)."""
    return rearrange(tok, "... (h w) (c ph pw) -> ... c (h ph) (w pw)", h=GRID, w=GRID, ph=PATCH, pw=PATCH)


def rope_freqs(ids: Tensor, axes: tuple[int, ...] = ROPE_AXES, theta: float = ROPE_THETA) -> Tensor:
    """(N, 3) ids → (N, HEAD/2, 2) (cos, sin) rotations, one frequency block per axis."""
    parts = []
    for i, dim in enumerate(axes):
        omega = 1.0 / theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
        parts.append(ids[:, i, None].double() * omega[None])
    ang = torch.cat(parts, -1)
    return torch.stack([ang.cos(), ang.sin()], -1).float()


def apply_rope(x: Tensor, f: Tensor) -> Tensor:
    """Rotate consecutive pairs of x (…, N, HEAD) by f (N, HEAD/2, 2)."""
    x2 = x.float().reshape(*x.shape[:-1], -1, 2)
    c, s = f[..., 0], f[..., 1]
    out = torch.stack([x2[..., 0] * c - x2[..., 1] * s, x2[..., 0] * s + x2[..., 1] * c], -1)
    return out.flatten(-2).type_as(x)


def timestep_embedding(t: Tensor, dim: int = T_EMBED, max_period: float = 10_000.0) -> Tensor:
    """FLUX's sinusoid of 1000·t: (…,) → (…, dim), cos ‖ sin."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (1000.0 * t)[..., None].float() * freqs
    return torch.cat([args.cos(), args.sin()], -1)


# -- blocks (SPEC §4.1, §4.2) ---------------------------------------------------------------
class Block(nn.Module):
    """FLUX parallel single-stream block: attention ‖ SwiGLU, one gate, QK RMSNorm, no biases."""

    def __init__(self, d: int, mlp_ratio: float = 3.0) -> None:
        super().__init__()
        self.heads, self.hid = d // HEAD, int(d * mlp_ratio)
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.mlp_in = nn.Linear(d, 2 * self.hid, bias=False)
        self.attn_out = nn.Linear(d, d, bias=False)
        self.mlp_out = nn.Linear(self.hid, d, bias=False)
        self.qn = nn.RMSNorm(HEAD, eps=1e-6)
        self.kn = nn.RMSNorm(HEAD, eps=1e-6)

    def forward(self, x: Tensor, mod: tuple[Tensor, Tensor, Tensor], pe: Tensor) -> Tensor:
        shift, scale, gate = mod
        h = (1 + scale) * self.norm(x) + shift
        q, k, v = rearrange(self.qkv(h), "b n (three h e) -> three b h n e", three=3, h=self.heads)
        q, k = apply_rope(self.qn(q), pe), apply_rope(self.kn(k), pe)
        a = rearrange(F.scaled_dot_product_attention(q, k, v), "b h n e -> b n (h e)")
        g, u = self.mlp_in(h).chunk(2, -1)
        return x + gate * (self.attn_out(a) + self.mlp_out(F.silu(g) * u))


class LastLayer(nn.Module):
    """FLUX LastLayer: Linear(LN(x)·(1 + scale) + shift) with its own adaLN, no biases."""

    def __init__(self, d: int, out: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Linear(d, 2 * d, bias=False)
        self.lin = nn.Linear(d, out, bias=False)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.ada(F.silu(vec)).chunk(2, -1)
        return self.lin((1 + scale) * self.norm(x) + shift)


class WAM(nn.Module):
    """The world action model. `forward` returns (future-Frame velocity (B, K, 16, 8, 8) | None, action head (B, H, 2))."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.frame_in = nn.Linear(PATCH_DIM, d)  # shared by observed and future Frames
        self.state_in = nn.Linear(STATE_DIM, d)
        self.instr = nn.Embedding(N_INSTR, d)  # outer, inner, any, null
        self.act_in = nn.Linear(ACTION_DIM, d)
        self.t_in = nn.Sequential(nn.Linear(T_EMBED, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))
        self.mods = nn.ModuleDict({s: nn.Linear(d, 3 * d, bias=False) for s in STREAMS if s in cfg.stream_lengths})
        self.blocks = nn.ModuleList(Block(d, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.fut_out = LastLayer(d, PATCH_DIM) if cfg.video_stream else None
        self.act_out = LastLayer(d, ACTION_DIM)
        self.register_buffer("pe", rope_freqs(position_ids(cfg)), persistent=False)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.instr.weight, std=0.02)
        for m in self.t_in:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for lin in self.mods.values():  # adaLN-Zero: every block starts as the identity
            nn.init.zeros_(lin.weight)
        for last in (self.fut_out, self.act_out):
            if last is not None:  # initial velocity exactly 0
                nn.init.zeros_(last.ada.weight)
                nn.init.zeros_(last.lin.weight)

    @property
    def streams(self) -> list[str]:
        return list(self.cfg.stream_lengths)

    def embed(self, obs: Tensor, state: Tensor, instr: Tensor, x_fut: Tensor | None, x_act: Tensor,
              t: Tensor) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Tokens x (B, N, d), per-token vec (B, N, d) and per-token (shift, scale, gate)."""
        cfg = self.cfg
        B = obs.shape[0]
        lens = cfg.stream_lengths
        toks = [self.frame_in(patchify(obs)), self.state_in(state)[:, None], self.instr(instr)[:, None]]
        if cfg.video_stream:
            if x_fut is None or tuple(x_fut.shape[1:]) != (cfg.n_future, 16, 8, 8):
                raise ValueError(f"x_fut must be (B, {cfg.n_future}, 16, 8, 8)")
            toks.append(self.frame_in(patchify(x_fut).flatten(1, 2)))
        toks.append(self.act_in(x_act))
        x = torch.cat(toks, 1)
        t = t.to(x.dtype)
        clean = lens["obs"] + lens["state"] + lens["instr"]
        noised = x.shape[1] - clean
        t_tok = torch.cat([torch.zeros(B, clean, device=x.device, dtype=x.dtype), t[:, None].expand(B, noised)], 1)
        vec = self.t_in(timestep_embedding(t_tok))
        mod = torch.cat([self.mods[s](F.silu(v)) for s, v in zip(lens, vec.split(list(lens.values()), 1))], 1)
        return x, vec, mod.chunk(3, -1)

    def trunk(self, x: Tensor, mod: tuple[Tensor, Tensor, Tensor], pe: Tensor | None = None) -> Tensor:
        pe = self.pe if pe is None else pe
        for b in self.blocks:
            x = b(x, mod, pe)
        return x

    def heads(self, x: Tensor, vec: Tensor) -> tuple[Tensor | None, Tensor]:
        lens = self.cfg.stream_lengths
        xs = dict(zip(lens, x.split(list(lens.values()), 1)))
        vs = dict(zip(lens, vec.split(list(lens.values()), 1)))
        v_act = self.act_out(xs["act"], vs["act"])
        v_fut = None
        if self.fut_out is not None:
            out = self.fut_out(xs["fut"], vs["fut"])
            v_fut = unpatchify(rearrange(out, "b (k n) p -> b k n p", n=FRAME_TOKENS))
        return v_fut, v_act

    def forward(self, obs: Tensor, state: Tensor, instr: Tensor, x_fut: Tensor | None, x_act: Tensor,
                t: Tensor) -> tuple[Tensor | None, Tensor]:
        """obs (B,16,8,8) · state (B,5) · instr (B,) long · x_fut (B,K,16,8,8) | None · x_act (B,H,2) · t (B,)."""
        x, vec, mod = self.embed(obs, state, instr, x_fut, x_act, t)
        return self.heads(self.trunk(x, mod), vec)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build(cfg: ModelConfig, seed: int | None = None) -> WAM:
    """A freshly initialised WAM; `seed` seeds torch's global RNG first (SPEC §6.4)."""
    if seed is not None:
        torch.manual_seed(seed)
    return WAM(cfg)
