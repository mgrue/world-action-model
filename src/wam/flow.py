"""Flow matching for the world action model (SPEC §5, §7): pure maths, no I/O.

Noise level t: x_t = t·ε + (1 − t)·x, target velocity ε − x; t = 1 is pure noise.
All random draws come from a CPU `torch.Generator` and are then moved to the device (SPEC §6.4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import ACTION_DIM, NULL, ModelConfig

HI_T = 0.7  # loss_action_hi_t: t > 0.7


# -- action normalisation (SPEC §5.1) -------------------------------------------------------
def _stats(cfg: ModelConfig, like: Tensor) -> tuple[Tensor, Tensor]:
    m = torch.tensor(cfg.action_mean, dtype=like.dtype, device=like.device)
    s = torch.tensor(cfg.action_std, dtype=like.dtype, device=like.device)
    return m, s


def normalise(actions: Tensor, cfg: ModelConfig) -> Tensor:
    m, s = _stats(cfg, actions)
    return (actions - m) / s


def denormalise(a_n: Tensor, cfg: ModelConfig) -> Tensor:
    m, s = _stats(cfg, a_n)
    return a_n * s + m


# -- noise levels (SPEC §5.2, §7) -----------------------------------------------------------
def timeshift(a: float, t: Tensor | float) -> Tensor | float:
    """a·t / (1 + (a − 1)·t): tilts t toward 1 (high noise) for a > 1."""
    return a * t / (1 + (a - 1) * t)


def sample_t(n: int, shift: float, generator: torch.Generator | None = None) -> Tensor:
    """Logit-normal t with the shift: u ~ N(0, 1), t = timeshift(shift, σ(u)); CPU, (n,)."""
    return timeshift(shift, torch.sigmoid(torch.randn(n, generator=generator)))


def schedule(n_steps: int, shift: float) -> Tensor:
    """The sampler's n_steps + 1 noise levels from 1 to 0 (float64, CPU)."""
    return timeshift(shift, torch.linspace(1, 0, n_steps + 1, dtype=torch.float64))


def noised(x: Tensor, eps: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """(x_t, target) for clean x, noise ε and per-sample t (B,)."""
    tt = t.view(-1, *([1] * (x.dim() - 1))).to(x.dtype)
    return tt * eps + (1 - tt) * x, eps - x


# -- the training draw and loss (SPEC §5.3–§5.6) --------------------------------------------
@dataclass
class Draw:
    """One batch's random quantities: t (B,), Null-Instruction mask (B,), action and video noise."""

    t: Tensor
    uncond: Tensor
    eps_a: Tensor
    eps_z: Tensor | None

    def to(self, device) -> Draw:
        return Draw(self.t.to(device), self.uncond.to(device), self.eps_a.to(device),
                    None if self.eps_z is None else self.eps_z.to(device))


def draw(cfg: ModelConfig, B: int, generator: torch.Generator | None = None, *,
         p_uncond: float | None = None) -> Draw:
    """Draw in a fixed order on the CPU: t, the p_uncond mask, action noise, video noise."""
    p = cfg.p_uncond if p_uncond is None else p_uncond
    t = sample_t(B, cfg.t_shift, generator)
    uncond = torch.rand(B, generator=generator) < p
    eps_a = torch.randn(B, cfg.horizon, ACTION_DIM, generator=generator)
    eps_z = torch.randn(B, cfg.n_future, 16, 8, 8, generator=generator) if cfg.video_stream else None
    return Draw(t, uncond, eps_a, eps_z)


def loss(model: Callable, cfg: ModelConfig, batch: dict, d: Draw) -> tuple[Tensor, dict[str, Tensor]]:
    """Weighted sum of per-Stream means (SPEC §5.3) or the regression loss (§5.6).

    `batch`: obs (B,16,8,8), fut (B,K,16,8,8) | None, state (B,5), instr (B,) long, actions (B,H,2) raw.
    `d` must be on the batch's device. Returns (loss, parts) where parts holds detached tensors:
    loss_video, loss_action (scalars) and per-Window action losses `action_per` (B,) and `t` (B,)."""
    obs, state, actions = batch["obs"], batch["state"], batch["actions"]
    instr = torch.where(d.uncond, torch.full_like(batch["instr"], NULL), batch["instr"])
    a_n = normalise(actions, cfg)
    B = obs.shape[0]
    if cfg.objective == "regression":
        ones = torch.ones(B, device=obs.device, dtype=obs.dtype)
        _, head = model(obs, state, instr, None, torch.zeros_like(a_n), ones)
        per = ((head - a_n) ** 2).mean((1, 2))
        la = per.mean()
        total = cfg.action_loss_weight * la
        return total, {"loss_video": None, "loss_action": la.detach(), "action_per": per.detach(), "t": ones}

    x_a, tgt_a = noised(a_n, d.eps_a, d.t)
    x_z = tgt_z = None
    if cfg.video_stream:
        x_z, tgt_z = noised(batch["fut"], d.eps_z, d.t)
    v_z, v_a = model(obs, state, instr, x_z, x_a, d.t)
    per = ((v_a - tgt_a) ** 2).mean((1, 2))
    la = per.mean()
    total = cfg.action_loss_weight * la
    lv = None
    if cfg.video_stream:
        lv = ((v_z - tgt_z) ** 2).mean()
        total = total + cfg.video_loss_weight * lv
    return total, {"loss_video": None if lv is None else lv.detach(), "loss_action": la.detach(),
                   "action_per": per.detach(), "t": d.t.detach()}


# -- sampling (SPEC §7) ---------------------------------------------------------------------
def initial_noise(cfg: ModelConfig, B: int, generator: torch.Generator | None = None) -> tuple[Tensor | None, Tensor]:
    """(x_z, x_a) pure noise on the CPU, drawn in this order (x_z only with video)."""
    x_z = torch.randn(B, cfg.n_future, 16, 8, 8, generator=generator) if cfg.video_stream else None
    x_a = torch.randn(B, cfg.horizon, ACTION_DIM, generator=generator)
    return x_z, x_a


def check_noise(cfg: ModelConfig, B: int, noise: tuple[Tensor | None, Tensor]) -> tuple[Tensor | None, Tensor]:
    """Validate caller-supplied initial noise against `initial_noise(cfg, B)`'s shapes."""
    try:
        x_z, x_a = noise
    except (TypeError, ValueError):
        raise ValueError("noise must be a pair (x_z, x_a)") from None
    want_a = (B, cfg.horizon, ACTION_DIM)
    if x_a is None or tuple(x_a.shape) != want_a:
        raise ValueError(f"action noise must have shape {want_a}, got {None if x_a is None else tuple(x_a.shape)}")
    if cfg.video_stream:
        want_z = (B, cfg.n_future, 16, 8, 8)
        if x_z is None or tuple(x_z.shape) != want_z:
            raise ValueError(f"video noise must have shape {want_z}, got {None if x_z is None else tuple(x_z.shape)}")
    elif x_z is not None:
        raise ValueError("video noise given for a model without the video Stream")
    return x_z, x_a


def guided(model: Callable, cfg_action: float = 1.0, cfg_video: float = 1.0) -> Callable:
    """model(obs, state, instr, x_z, x_a, t) with per-Stream classifier-free guidance:
    v = v_uc + w·(v_c − v_uc), v_uc with the Null Instruction, in one batched forward of 2B.
    At w = 1 for both Streams it is the model itself (one pass of B)."""
    if cfg_action == 1.0 and cfg_video == 1.0:
        return model

    def fn(obs, state, instr, x_z, x_a, t):
        B = obs.shape[0]
        cat = lambda x: None if x is None else torch.cat([x, x])  # noqa: E731
        vz, va = model(cat(obs), cat(state), torch.cat([instr, torch.full_like(instr, NULL)]), cat(x_z), cat(x_a),
                       cat(t))
        va = va[B:] + cfg_action * (va[:B] - va[B:])
        vz = None if vz is None else vz[B:] + cfg_video * (vz[:B] - vz[B:])
        return vz, va

    return fn


def euler(model: Callable, obs: Tensor, state: Tensor, instr: Tensor, x_z: Tensor | None, x_a: Tensor, *,
          n_steps: int, t_shift: float) -> tuple[Tensor | None, Tensor]:
    """Plain Euler from t = 1 to 0 over `schedule(n_steps, t_shift)`; one model call per step."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥ 1, got {n_steps}")
    ts = schedule(n_steps, t_shift).tolist()
    B = obs.shape[0]
    for t, tn in zip(ts[:-1], ts[1:]):
        tt = torch.full((B,), t, device=obs.device, dtype=obs.dtype)
        v_z, v_a = model(obs, state, instr, x_z, x_a, tt)
        x_a = x_a + (tn - t) * v_a
        if x_z is not None:
            x_z = x_z + (tn - t) * v_z
    return x_z, x_a


@torch.no_grad()
def sample(model: Callable, cfg: ModelConfig, obs: Tensor, state: Tensor, instr: Tensor, *, n_steps: int = 10,
           cfg_action: float = 1.0, cfg_video: float = 1.0, generator: torch.Generator | None = None,
           noise: tuple[Tensor | None, Tensor] | None = None) -> tuple[Tensor | None, Tensor]:
    """(future Latents (B,K,16,8,8) normalised | None, actions (B,H,2) normalised).

    `noise = (x_z, x_a)` as `initial_noise` returns it replaces the draw from `generator` (Step 9 SPEC §11);
    giving both, or a wrong shape, raises ValueError. Ignored by the regression objective."""
    B, dev = obs.shape[0], obs.device
    if noise is not None and generator is not None:
        raise ValueError("pass either noise or generator, not both")
    if cfg.objective == "regression":
        if cfg_action != 1.0 or cfg_video != 1.0:
            raise ValueError("classifier-free guidance is not defined for objective=regression")
        ones = torch.ones(B, device=dev, dtype=obs.dtype)
        _, a = model(obs, state, instr, None, torch.zeros(B, cfg.horizon, ACTION_DIM, device=dev), ones)
        return None, a
    x_z, x_a = initial_noise(cfg, B, generator) if noise is None else check_noise(cfg, B, noise)
    x_z = None if x_z is None else x_z.to(dev)
    return euler(guided(model, cfg_action, cfg_video), obs, state, instr, x_z, x_a.to(dev),
                 n_steps=n_steps, t_shift=cfg.t_shift)
