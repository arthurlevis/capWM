"""Collect finger/cylinder trajectories in the simulator with a random pushing policy.

Used for (a) the real-to-sim check: the same world model evaluated on sim
transitions, (b) a sim-only ablation of the world model, and (c) friction tuning.

Usage: python scripts/collect_sim_data.py --episodes 100 --out data/processed/sim
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from tqdm import tqdm

from pushwm.config import load_config
from pushwm.data.trajectories import Trajectory
from pushwm.sim import PushEnv


def random_push_episode(env: PushEnv, rng: np.random.Generator, steps=60, max_speed=0.02, n_obstacles=0):
    """Alternate: approach the cylinder from a random side, push a random distance
    in a random direction at a random speed, wander freely. Mirrors the recording
    protocol: a person does not ram the cylinder into walls, so pushes that would
    end within a margin of a wall are re-drawn."""
    cfg = env.cfg
    obs = env.reset(n_obstacles=n_obstacles)
    r_std = cfg.cylinder.radius + cfg.probe.radius
    F, C, V = [obs.finger.copy()], [obs.cylinder.copy()], [True]
    mode, left = "approach", 0
    push_dir, speed, waypoint = None, 0.0, None
    for _ in range(steps):
        if mode == "approach":
            if waypoint is None:
                ang = rng.uniform(0, 2 * np.pi)
                waypoint = obs.cylinder + np.array([np.cos(ang), np.sin(ang)]) * (r_std + rng.uniform(0.005, 0.03))
                waypoint = np.clip(waypoint, 0.02, env.size - 0.02)
            delta = waypoint - obs.finger
            if np.linalg.norm(delta) < 0.004:
                mode = "push"; waypoint = None
                # Push roughly toward the cylinder with some angular noise, random speed.
                to_cyl = obs.cylinder - obs.finger
                for _ in range(20):
                    base = np.arctan2(to_cyl[1], to_cyl[0]) + rng.normal(0, 0.5)
                    push_dir = np.array([np.cos(base), np.sin(base)])
                    speed = rng.uniform(0.004, max_speed)
                    left = int(rng.integers(3, 12))
                    end = obs.cylinder + push_dir * speed * left
                    margin = cfg.cylinder.radius + 0.02
                    if np.all(end > margin) and np.all(end < env.size - margin):
                        break
                else:
                    mode = "wander"; left = int(rng.integers(3, 10)); waypoint = None; continue
        elif mode == "push":
            delta = push_dir * speed
            left -= 1
            if left <= 0:
                mode = "wander" if rng.random() < 0.4 else "approach"
                left = int(rng.integers(3, 10))
        else:  # wander: free motion that should not move the cylinder
            if left == int(rng.integers(3, 10)) or waypoint is None:
                waypoint = rng.uniform(0.02, env.size - 0.02)
            delta = waypoint - obs.finger
            if np.linalg.norm(delta) > max_speed:
                delta = delta / np.linalg.norm(delta) * rng.uniform(0.005, max_speed)
            left -= 1
            if left <= 0 or np.linalg.norm(delta) < 0.004:
                mode = "approach"; waypoint = None
        obs = env.step(delta)
        F.append(obs.finger.copy()); C.append(obs.cylinder.copy()); V.append(True)
    return Trajectory(np.array(F), np.array(C), np.array(V), cfg.control.dt, "sim", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/processed/sim")
    p.add_argument("--config", default=None)
    p.add_argument("--max-speed", type=float, default=0.02)
    p.add_argument("--obstacles", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config)
    env = PushEnv(cfg, seed=a.seed)
    rng = np.random.default_rng(a.seed)
    n_moving = 0; n_tot = 0
    for i in tqdm(range(a.episodes)):
        tr = random_push_episode(env, rng, a.steps, a.max_speed, a.obstacles)
        tr.name = f"sim_{a.seed:02d}_{i:04d}"
        tr.save(os.path.join(a.out, tr.name + ".npz"))
        d = np.linalg.norm(np.diff(tr.cylinder, axis=0), axis=1)
        n_moving += int((d > 0.002).sum()); n_tot += len(d)
    print(f"saved {a.episodes} episodes to {a.out}; cylinder moving in {n_moving}/{n_tot} transitions ({n_moving/n_tot:.0%})")
    env.close()


if __name__ == "__main__":
    main()
