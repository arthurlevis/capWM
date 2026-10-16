"""Retarget the recorded push clips into robot demonstrations for policy fine-tuning.

For each push clip: the tracked fingertip and cap trajectory (box metres, 10 Hz) is
replayed kinematically in the simulator, the probe marker standing in for the
finger, the goal placed where the cap ended, obstacles where the red houses were
detected in the clip's background. Each step yields the simulator's overhead
image (arm-free), state = probe xy, action = the recorded fingertip displacement.
Every motion is the recording's; every pixel is the simulator's.

    python scripts/retarget_demos.py --videos data/raw --traj data/processed/video --out data/demos_retargeted
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import numpy as np
from tqdm import tqdm

from pushwm.config import load_config
from pushwm.data.trajectories import Trajectory
from pushwm.sim import PushEnv
from pushwm.vision.calib import BoxCalibration, median_background
from pushwm.vision.track import red_obstacle_mask

TASK = "Push the blue cap to the green target without touching the red obstacles."


def detect_obstacles(bg, calib, r_ob, min_area=150):
    mask = red_obstacle_mask(bg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        M = cv2.moments(c)
        xy = calib.to_box([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        if np.all(xy > 0) and np.all(xy < [calib.size_x, calib.size_y]):
            out.append([xy[0], xy[1], r_ob])
    return np.array(out, dtype=np.float32).reshape(-1, 3)


def longest_valid_run(valid):
    best, cur = (0, 0), None
    for i, v in enumerate(valid):
        if v and cur is None:
            cur = i
        if (not v or i == len(valid) - 1) and cur is not None:
            end = i + 1 if v else i
            if end - cur > best[1] - best[0]:
                best = (cur, end)
            cur = None
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="data/raw")
    p.add_argument("--traj", default="data/processed/video")
    p.add_argument("--calib", default="data/calib/box.json")
    p.add_argument("--out", default="data/demos_retargeted")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--min-travel", type=float, default=0.03, help="cap must travel at least this far (m) in the usable window")
    p.add_argument("--max-step", type=float, default=0.02)
    a = p.parse_args()
    cfg = load_config(); calib = BoxCalibration.load(a.calib)
    env = PushEnv(cfg, seed=0)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    index, skipped = [], []
    for npz in tqdm(sorted(Path(a.traj).glob("IMG_*.npz"))):
        tr = Trajectory.load(npz)
        s, e = longest_valid_run(tr.valid)
        if e - s < 5:
            skipped.append((tr.name, "too short")); continue
        F, C = tr.finger[s:e], tr.cylinder[s:e]
        if np.linalg.norm(C[-1] - C[0]) < a.min_travel:
            skipped.append((tr.name, "cap barely moved (no-contact clip)")); continue
        bg = median_background(Path(a.videos) / f"{tr.name}.MOV", 15)
        obstacles = detect_obstacles(bg, calib, cfg.obstacles.radius)
        goal = C[-1].copy()
        env.reset(start=list(C[0]), goal=list(goal), obstacles=obstacles if len(obstacles) else None, n_obstacles=0 if not len(obstacles) else None)
        imgs, states, actions, caps = [], [], [], []
        for t in range(len(F) - 1):
            env.replay_pose(F[t], C[t])
            act = np.clip(F[t + 1] - F[t], -a.max_step, a.max_step).astype(np.float32)
            imgs.append(env.render_overhead(a.size)); states.append(F[t].astype(np.float32)); actions.append(act); caps.append(C[t].astype(np.float32))
        env.end_replay()
        name = f"demo_{tr.name}"
        np.savez_compressed(out / f"{name}.npz", images=np.stack(imgs), state=np.stack(states), action=np.stack(actions),
                            cap=np.stack(caps), goal=goal.astype(np.float32), obstacles=obstacles, layout="video")
        index.append(dict(name=name, clip=tr.name, steps=len(actions), n_obstacles=int(len(obstacles)), travel_cm=float(np.linalg.norm(C[-1] - C[0]) * 100)))
    json.dump(dict(task=TASK, fps=int(round(1 / cfg.control.dt)), image_size=a.size, source="recorded phone clips, kinematically replayed in the simulator",
                   state="probe xy (m, box frame)", action=f"recorded fingertip displacement per step (m), clipped to +-{a.max_step}",
                   episodes=index, skipped=skipped), open(out / "meta.json", "w"), indent=1)
    print(f"{len(index)} demonstrations, {sum(i['steps'] for i in index)} frames; skipped {len(skipped)}: {[n for n, _ in skipped][:20]}")


if __name__ == "__main__":
    main()
