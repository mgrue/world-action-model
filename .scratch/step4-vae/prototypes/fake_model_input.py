"""Explainer, not a ticket answer: one batch in the shape the main model (Step 6) would receive.

Real Windows from data/step3-v1 (val eval_subset), but FAKE latents, because there is no
encoder yet. Each fake latent cell (8×8 pixels = 5×5 m) holds:
    channel 0–2   mean colour of the cell, scaled to about [-1, 1]
    channel 3     share of the cell covered by the car (0 … 1), so you can find the car
    channel 4–15  small seeded noise (a real encoder would use these freely)
A real encoder learns its own channel meanings; only the SHAPES here are the point.

    uv run python .scratch/step4-vae/prototypes/fake_model_input.py

Writes prototypes/out/fake_model_input.npz and prototypes/out/fake_model_input.txt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from toycar import State, render
from toycar.dataset import open_dataset

B = 4  # windows in the batch
H = 16  # future steps
C, G = 16, 8  # latent channels, latent grid size (8×8)
TYPES = ("obs_latent", "state", "instruction", "future_latent", "action")
OUT = Path(__file__).parent / "out"


def fake_encode(frame: np.ndarray, background: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(64, 64, 3) uint8 → (16, 8, 8) float32. Stand-in for the Step 4 encoder."""
    cells = frame.reshape(G, 8, G, 8, 3).astype(np.float32)
    colour = cells.mean(axis=(1, 3)) / 127.5 - 1.0  # (8, 8, 3)
    car = (np.abs(frame.astype(int) - background.astype(int)).sum(-1) > 0).reshape(G, 8, G, 8).mean(axis=(1, 3))
    z = np.empty((C, G, G), np.float32)
    z[0:3] = colour.transpose(2, 0, 1)
    z[3] = car
    z[4:] = 0.1 * rng.standard_normal((C - 4, G, G))
    return z


def main() -> None:
    ds = open_dataset("data/step3-v1", check=False)
    es = ds.split("val")
    w = es.eval_subset()
    w = w.subset(np.flatnonzero(w.approaches_section)[:2].tolist() + [0, 1])  # 2 near the fork, 2 others
    rng = np.random.default_rng(0)

    # -- 1. raw per-Window data, as the Step 3 loader gives it -------------------------------
    items = [w[i] for i in range(B)]
    obs_latents = np.empty((B, C, G, G), np.float32)
    future_clean = np.empty((B, H, C, G, G), np.float32)
    for b, it in enumerate(items):
        rp = es.render_params(int(it["episode"]))
        bg = render(State(-100.0, -100.0, 0.0, 0.0), es.track, rp)  # same colours, car off-screen
        obs_latents[b] = fake_encode(it["obs_frame"], bg, rng)
        for k in range(H):
            future_clean[b, k] = fake_encode(it["future_frames"][k], bg, rng)
    state = np.stack([it["state"] for it in items])  # (B, 5)
    instruction = np.array([int(it["instruction"]) for it in items], np.int64)  # (B,)
    actions_clean = np.stack([it["actions"] for it in items])  # (B, 16, 2)

    # -- 2. flow-matching noising (Step 7): one noise level t per Window ---------------------
    t = rng.uniform(0.0, 1.0, B).astype(np.float32)
    noise_lat = rng.standard_normal(future_clean.shape).astype(np.float32)
    noise_act = rng.standard_normal(actions_clean.shape).astype(np.float32)
    tl, ta = t[:, None, None, None, None], t[:, None, None]
    future_noisy = tl * noise_lat + (1 - tl) * future_clean
    actions_noisy = ta * noise_act + (1 - ta) * actions_clean
    target_v_lat = noise_lat - future_clean  # what the model should output for future latents
    target_v_act = noise_act - actions_clean  # … and for actions

    # -- 3. pack into one token sequence (Step 6) --------------------------------------------
    # Each type has its own width (16, 5, 1, 16, 2). A real model projects each type to width D
    # with its own layer; here we pad to 16 so it fits in one array, with a width per token.
    rows, cols = np.divmod(np.arange(G * G), G)
    meta = []  # (type, time, row, col, noisy, width)
    meta += [(0, 0, r, c, 0, C) for r, c in zip(rows, cols)]  # observed Frame, time 0
    meta += [(1, 0, -1, -1, 0, 5)]  # state, time 0
    meta += [(2, 0, -1, -1, 0, 1)]  # instruction
    meta += [(3, k + 1, r, c, 1, C) for k in range(H) for r, c in zip(rows, cols)]  # future Frame k at time k+1
    meta += [(4, k + 1, -1, -1, 1, 2) for k in range(H)]  # action k at time k+1 (same as Frame k)
    meta_a = np.array(meta, np.int64)
    N = len(meta_a)
    tokens = np.zeros((B, N, C), np.float32)
    o = 0
    tokens[:, o : o + 64] = obs_latents.reshape(B, C, 64).transpose(0, 2, 1); o += 64
    tokens[:, o, :5] = state; o += 1
    tokens[:, o, 0] = instruction; o += 1
    tokens[:, o : o + H * 64] = future_noisy.reshape(B, H, C, 64).transpose(0, 1, 3, 2).reshape(B, H * 64, C); o += H * 64
    tokens[:, o : o + H, :2] = actions_noisy; o += H
    assert o == N

    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "fake_model_input.npz",
        # raw pieces
        obs_latents=obs_latents, state=state, instruction=instruction,
        future_latents_clean=future_clean, actions_clean=actions_clean,
        # noising
        t=t, future_latents_noisy=future_noisy, actions_noisy=actions_noisy,
        target_velocity_latents=target_v_lat, target_velocity_actions=target_v_act,
        # packed sequence
        tokens=tokens, token_type=meta_a[:, 0], token_time=meta_a[:, 1], token_row=meta_a[:, 2],
        token_col=meta_a[:, 3], token_noisy=meta_a[:, 4].astype(bool), token_width=meta_a[:, 5],
        # traceability
        window_episode=np.array([int(i["episode"]) for i in items]), window_t=np.array([int(i["t"]) for i in items]),
    )

    # -- human-readable dump -----------------------------------------------------------------
    L = []
    L.append("FAKE model input — real Windows from data/step3-v1 (val), fake latents (no encoder yet)\n")
    L.append(f"batch B = {B} Windows, horizon H = {H}, latent = {C} channels × {G}×{G} grid\n")
    L.append("== 1. Raw pieces per Window (shape includes the batch B) ==")
    with np.load(OUT / "fake_model_input.npz") as z:
        for k in z.files:
            L.append(f"  {k:26s} {str(z[k].shape):22s} {z[k].dtype}")
    L.append("\n== Windows in this batch ==")
    for b, it in enumerate(items):
        L.append(f"  [{b}] episode {int(it['episode'])}, t {int(it['t'])}, instruction "
                 f"{['outer', 'inner', 'any'][int(it['instruction'])]}, approaches_section {bool(it['approaches_section'])}, "
                 f"noise level t = {t[b]:.2f}")
    b = 0
    L.append("\n== Window 0: state token (x/40, y/40, cos θ, sin θ, v/10) ==")
    L.append("  " + "  ".join(f"{v:+.3f}" for v in state[b]))
    L.append("\n== Window 0: observed-Frame fake latent, channel 3 = car coverage per cell (find the car) ==")
    for r in range(G):
        L.append("  " + " ".join(f"{v:4.2f}" if v > 0 else "  . " for v in obs_latents[b, 3, r]))
    L.append("\n== Window 0: same channel in future Frame k = 15 (clean) — the car has moved ==")
    for r in range(G):
        L.append("  " + " ".join(f"{v:4.2f}" if v > 0 else "  . " for v in future_clean[b, 15, 3, r]))
    L.append("\n== Window 0: actions — clean, noisy (model input), target velocity (model output) ==")
    L.append("   k  time   clean steer/thr     noisy steer/thr    target v steer/thr")
    for k in range(H):
        L.append(f"  {k:2d}  t+{k + 1:<3d} {actions_clean[b, k, 0]:+.3f} {actions_clean[b, k, 1]:+.3f}     "
                 f"{actions_noisy[b, k, 0]:+.3f} {actions_noisy[b, k, 1]:+.3f}     "
                 f"{target_v_act[b, k, 0]:+.3f} {target_v_act[b, k, 1]:+.3f}")
    L.append(f"\n== 3. Packed token sequence: tokens {tokens.shape} (B, N tokens, padded width 16) ==")
    counts = np.bincount(meta_a[:, 0], minlength=len(TYPES))
    for i, name in enumerate(TYPES):
        L.append(f"  {name:14s} {counts[i]:5d} tokens, width {meta_a[meta_a[:, 0] == i][0, 5]:2d}, "
                 f"{'noisy' if meta_a[meta_a[:, 0] == i][0, 4] else 'clean'}")
    L.append(f"  {'total':14s} {N:5d} tokens\n")
    L.append("  Selected rows of Window 0 (idx, type, time, row, col, noisy, first values):")
    show = list(range(0, 3)) + [63, 64, 65, 66, 67, 66 + 64 * 15, N - 17, N - 16, N - 1]
    for i in show:
        ty, tm, r, c, nz, wd = meta_a[i]
        vals = " ".join(f"{v:+.2f}" for v in tokens[b, i, : min(wd, 5)])
        pos = f"row {r} col {c}" if r >= 0 else "          "
        L.append(f"  {i:5d}  {TYPES[ty]:14s} time {tm:2d}  {pos}  {'noisy' if nz else 'clean'}  [{vals}{' …' if wd > 5 else ''}]")
    L.append("\n  Next in a real model: per-type input layers map each token to width D, position tags")
    L.append("  (time, row, col) are added, t goes in through adaptive layer norm, and only the noisy")
    L.append("  tokens (future latents, actions) get output heads, trained towards the target velocities.")
    (OUT / "fake_model_input.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
