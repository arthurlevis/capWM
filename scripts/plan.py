"""Run the world-model planner in the simulator on random episodes.

    python scripts/plan.py --model outputs/wm_video.pt --episodes 20 --video outputs/plan_demo.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from pushwm.config import load_config
from pushwm.models.state_wm import StateWorldModel
from pushwm.planning.cem import CEMPlanner, PlannerConfig
from pushwm.sim import PushEnv


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = StateWorldModel(ck["n_models"], ck["hidden"], ck["depth"], ck.get("history", 1))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def run_episode(env, planner, max_steps=120, frames=None, reset_kwargs=None):
    obs = env.reset(**(reset_kwargs or {}))
    planner.reset()
    log = [obs.as_dict()]
    if frames is not None:
        frames.append(np.concatenate([env.render("top"), env.render("side")], axis=1))
    for _ in range(max_steps):
        a = planner.plan(obs.finger, obs.cylinder, obs.goal, obs.obstacles)
        log[-1]["pred_cyl"] = planner.last_info["pred_cyl"].copy()      # plan made from this state
        log[-1]["pred_finger"] = planner.last_info["pred_finger"].copy()
        prev = obs
        obs = env.step(a)
        planner.observe_transition(obs.cylinder - prev.cylinder, obs.finger - prev.finger)
        log.append(obs.as_dict())
        if frames is not None:
            frames.append(np.concatenate([env.render("top"), env.render("side")], axis=1))
        if obs.success or obs.collision:
            break
    return log


def summarize(log, obstacles):
    first, last = log[0], log[-1]
    path = sum(np.linalg.norm(log[i + 1]["cylinder"] - log[i]["cylinder"]) for i in range(len(log) - 1))
    min_clear = np.inf
    for rec in log:
        for ob in obstacles:
            min_clear = min(min_clear, np.linalg.norm(rec["cylinder"] - ob[:2]) - ob[2])
    collided = any(bool(r.get("collision", False)) for r in log)
    success = bool(last["success"]) and not collided
    return dict(success=success, collided=collided, timeout=(not success and not collided), steps=len(log) - 1,
                final_dist=float(np.linalg.norm(last["cylinder"] - last["goal"])),
                start_dist=float(np.linalg.norm(first["cylinder"] - first["goal"])),
                path_len=float(path), min_obstacle_clearance=float(min_clear) if np.isfinite(min_clear) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="world-model checkpoint (required with --policy wm)")
    p.add_argument("--policy", choices=["wm", "rule"], default="wm",
                   help="wm: learned world model; rule: hand-written rigid-contact model in the same planner")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=80, help="time limit in control steps (0.1 s each); exceeding it is a failure")
    p.add_argument("--video", default=None, help="record the first --video-episodes episodes to this mp4")
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--video-dir", default=None, help="write one mp4 per episode (Panda top + side view) into this folder")
    p.add_argument("--video-scale", type=float, default=0.5, help="downscale factor for per-episode mp4s")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--w-std", type=float, default=2.0)
    p.add_argument("--no-geodesic", action="store_true", help="straight-line goal distance (ablation)")
    p.add_argument("--clearance", type=float, default=None, help="planner obstacle clearance margin in metres (default from PlannerConfig)")
    p.add_argument("--obstacles", type=int, default=None, help="fixed obstacle count (default random)")
    p.add_argument("--out", default=None, help="json results path")
    p.add_argument("--hard", action="store_true", help="three-obstacle barrier across the start-goal line")
    p.add_argument("--opposite", action="store_true", help="chessboard layout: start and goal in opposite edge zones, 4-6 obstacles between")
    p.add_argument("--save-logs", default=None, help="npz with per-step states and plans for rendering")
    p.add_argument("--config", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    pcfg = PlannerConfig(horizon=a.horizon, n_samples=a.samples, n_iters=a.iters, w_std=a.w_std,
                         max_step=min(cfg.control.max_step, 0.02), seed=a.seed, geodesic=not a.no_geodesic)
    if a.clearance is not None:
        pcfg.clearance = a.clearance
    env = PushEnv(cfg, seed=a.seed)
    if a.policy == "wm":
        model = load_model(a.model, pcfg.device)
        planner = CEMPlanner(model, cfg, pcfg)
    else:
        from pushwm.models.rule_wm import RuleWorldModel
        model = RuleWorldModel(cfg.cylinder.radius + cfg.probe.radius, pcfg.device)
        planner = CEMPlanner(model, cfg, pcfg)
    results, frames, logs = [], ([] if a.video else None), []
    reset_kwargs = dict(n_obstacles=a.obstacles, obstacle_on_line=a.hard, opposite_sides=a.opposite)
    t0 = time.time()
    if a.video_dir:
        os.makedirs(a.video_dir, exist_ok=True)
    for ep in range(a.episodes):
        rec = frames if (frames is not None and ep < a.video_episodes) else None
        if a.video_dir:
            rec = []
        log = run_episode(env, planner, a.max_steps, rec, reset_kwargs)
        logs.append(log)
        if a.video_dir and rec:
            collided = any(bool(r.get("collision", False)) for r in log)
            ok = bool(log[-1]["success"]) and not collided
            tag = "success" if ok else ("fail_collision" if collided else "fail_timeout")
            small = [cv2.resize(f, None, fx=a.video_scale, fy=a.video_scale, interpolation=cv2.INTER_AREA) for f in rec]
            imageio.mimwrite(os.path.join(a.video_dir, f"episode_{ep:03d}_{tag}.mp4"),
                             small + [small[-1]] * 5, fps=int(1 / cfg.control.dt), quality=6)
            if frames is not None and ep < a.video_episodes:
                frames.extend(rec)
        s = summarize(log, log[0]["obstacles"]); results.append(s)
        print(f"ep {ep:2d}: success={s['success']} collided={s['collided']} steps={s['steps']:3d} final={s['final_dist']*100:.1f}cm "
              f"start={s['start_dist']*100:.1f}cm path={s['path_len']*100:.1f}cm clear={s['min_obstacle_clearance']}")
    succ = np.mean([r["success"] for r in results])
    print(f"success rate {succ:.0%} over {len(results)} episodes | {time.time()-t0:.0f}s")
    if a.video and frames:
        imageio.mimwrite(a.video, frames, fps=int(1 / cfg.control.dt), quality=7)
        print("video", a.video)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(results=results, success_rate=float(succ), args=vars(a)), f, indent=1)
    if a.save_logs:
        np.save(a.save_logs, np.array(logs, dtype=object), allow_pickle=True)
    env.close()


if __name__ == "__main__":
    main()
