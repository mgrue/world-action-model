"""Keyboard driving mode (SPEC §7).

    uv run python -m toycar.drive [--seed N] [--direction cw|ccw] [--route outer|inner] [--speed V] [--record DIR]

Keys: ↑/W throttle, ↓/S brake, ←/A →/D steer, H hi-res, Tab panel, P pause,
R next seed, Shift+R same seed, Esc/Q quit.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence

import numpy as np
import pygame

from .env import CarEnv
from .physics import DT
from .render import render
from .replay import Recording
from .state import StartSpec

SCALE = 8
VIEW = 64 * SCALE
PANEL_W = 220
STEER_RATE = 4.0  # /s toward ±1 while a steer key is held
STEER_RETURN = 6.0  # /s back toward 0 when released
CRASH_FREEZE_S = 1.5


def ramp(value: float, target: float, rate: float) -> float:
    step = rate * DT
    if value < target:
        return min(value + step, target)
    return max(value - step, target)


def read_action(keys, steer: float) -> tuple[float, float]:
    up = keys[pygame.K_UP] or keys[pygame.K_w]
    down = keys[pygame.K_DOWN] or keys[pygame.K_s]
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    throttle = 1.0 if up and not down else -1.0 if down and not up else 0.0
    if left and not right:
        steer = ramp(steer, -1.0, STEER_RATE)
    elif right and not left:
        steer = ramp(steer, 1.0, STEER_RATE)
    else:
        steer = ramp(steer, 0.0, STEER_RETURN)
    return steer, throttle


class Driver:
    def __init__(self, seed: int, start: StartSpec, record_dir: str | None) -> None:
        self.env = CarEnv(max_steps=10**9)
        self.start = start
        self.record_dir = record_dir
        self.seed = seed
        self.steer = 0.0
        self.throttle = 0.0
        self.recording: Recording | None = None
        self.info = None
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        print(f"seed {seed}", flush=True)
        state, self.info = self.env.reset(seed, self.start)
        self.steer = self.throttle = 0.0
        if self.record_dir is not None:
            self.recording = Recording(seed, self.start, self.env.render_params, self.env.track.params)
            self.recording.add(state, self.env.render())

    def end_episode(self, termination: str) -> None:
        if self.recording is not None and self.record_dir is not None:
            path = self.recording.save(self.record_dir, termination)
            print(f"recorded {len(self.recording.actions)} ticks -> {path} ({termination})", flush=True)
            self.recording = None

    def tick(self, keys) -> bool:
        """Apply one Action. Returns True on a crash."""
        self.steer, self.throttle = read_action(keys, self.steer)
        action = (self.steer, self.throttle)
        result = self.env.step(action)
        self.info = result.info
        if self.recording is not None:
            self.recording.add(result.state, self.env.render(), action)
        return result.terminated


def draw_panel(screen: pygame.Surface, font: pygame.font.Font, d: Driver, paused: bool, hires: bool) -> None:
    s, i = d.env.state, d.info
    lines = [
        f"x        {s.x:7.2f} m",
        f"y        {s.y:7.2f} m",
        f"theta    {math.degrees(s.theta):7.1f} deg",
        f"v        {s.v:7.2f} m/s",
        f"steer    {d.steer:+7.2f}",
        f"throttle {d.throttle:+7.2f}",
        f"station  {i.station_m:7.2f} m",
        f"progress {i.progress_m:7.2f} m",
        f"laps     {i.laps}",
        f"direction {i.direction}",
        f"branch   {i.branch}",
        f"last_br  {i.last_branch}",
        f"wrong_way {i.wrong_way}",
        f"steps    {i.steps}",
        f"seed     {d.seed}",
        "",
        ("PAUSED" if paused else "") + ("  HI-RES" if hires else ""),
        "",
        "Arrows/WASD drive",
        "H hi-res  Tab panel  P pause",
        "R next seed  Shift+R same",
        "Esc/Q quit",
    ]
    pygame.draw.rect(screen, (24, 24, 28), (VIEW, 0, PANEL_W, VIEW))
    for k, text in enumerate(lines):
        color = (255, 120, 120) if text.startswith("wrong_way True") else (220, 220, 220)
        screen.blit(font.render(text, True, color), (VIEW + 10, 10 + k * 20))


def frame_surface(frame: np.ndarray) -> pygame.Surface:
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return surf if surf.get_size() == (VIEW, VIEW) else pygame.transform.scale(surf, (VIEW, VIEW))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m toycar.drive", description="Drive the toycar with the keyboard.")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--direction", choices=["cw", "ccw"])
    ap.add_argument("--route", choices=["outer", "inner"])
    ap.add_argument("--speed", type=float, default=0.0)
    ap.add_argument("--record", metavar="DIR")
    args = ap.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    start = StartSpec(direction=args.direction, route=args.route, speed=args.speed)

    pygame.init()
    show_panel = True
    screen = pygame.display.set_mode((VIEW + PANEL_W, VIEW))
    pygame.display.set_caption("toycar drive")
    font = pygame.font.SysFont("menlo,monaco,couriernew,monospace", 15)
    banner_font = pygame.font.SysFont("helvetica,arial", 72, bold=True)
    clock = pygame.time.Clock()

    driver = Driver(seed, start, args.record)
    paused = hires = False
    crash_until: float | None = None
    try:
        while True:
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        quit_requested = True
                    elif event.key == pygame.K_h:
                        hires = not hires
                    elif event.key == pygame.K_TAB:
                        show_panel = not show_panel
                        screen = pygame.display.set_mode((VIEW + (PANEL_W if show_panel else 0), VIEW))
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r and crash_until is None:
                        driver.end_episode("reset")
                        same = bool(event.mod & pygame.KMOD_SHIFT)
                        driver.reset(driver.seed if same else driver.seed + 1)
            if quit_requested:
                if crash_until is None:
                    driver.end_episode("quit")
                break

            if crash_until is not None:
                if time.monotonic() >= crash_until:
                    crash_until = None
                    driver.reset(driver.seed + 1)
            elif not paused:
                if driver.tick(pygame.key.get_pressed()):
                    driver.end_episode("crash")
                    crash_until = time.monotonic() + CRASH_FREEZE_S

            env = driver.env
            if hires:
                frame = render(env.state, env.track, env.render_params, resolution=VIEW)
            else:
                frame = env.render()
            screen.blit(frame_surface(frame), (0, 0))
            if crash_until is not None:
                text = banner_font.render("CRASH", True, (255, 255, 255))
                box = text.get_rect(center=(VIEW // 2, VIEW // 2)).inflate(40, 20)
                pygame.draw.rect(screen, (200, 20, 20), box)
                screen.blit(text, text.get_rect(center=box.center))
            if show_panel:
                draw_panel(screen, font, driver, paused, hires)
            pygame.display.flip()
            clock.tick(15)
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
