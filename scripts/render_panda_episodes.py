"""Re-run a benchmark evaluation deterministically and render selected episodes from
the isometric camera (whole robot visible) as mp4 and gif.

    python scripts/render_panda_episodes.py --layout opposite --episodes 2 15 48 --out docs/media/opposite/world_planner/panda
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio.v2 as imageio
import numpy as np

from pushwm.config import load_config
from pushwm.models.state_wm import load_state_wm
from pushwm.planning.cem import CEMPlanner, PlannerConfig
from pushwm.sim import PushEnv

SEEDS = {"opposite": 600, "barrier": 700}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=["opposite", "barrier"], required=True)
    p.add_argument("--episodes", nargs="+", type=int, required=True)
    p.add_argument("--model", default="models/wm_video_h0.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--camera", default="iso")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--keep-mp4", action="store_true", help="keep the intermediate mp4 next to the gif")
    a = p.parse_args()
    cfg = load_config(); seed = SEEDS[a.layout]
    pcfg = PlannerConfig(seed=seed)
    planner = CEMPlanner(load_state_wm(a.model, pcfg.device), cfg, pcfg)
    env = PushEnv(cfg, seed=seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    want = set(a.episodes)
    reset_kwargs = dict(opposite_sides=(a.layout == "opposite"), obstacle_on_line=(a.layout == "barrier"))
    for ep in range(max(want) + 1):
        obs = env.reset(**reset_kwargs); planner.reset()
        frames = [] if ep in want else None
        collided = False
        if frames is not None:
            frames.append(env.render(a.camera, a.width, a.height))
        for t in range(a.max_steps):
            act = planner.plan(obs.finger, obs.cylinder, obs.goal, obs.obstacles)
            prev = obs; obs = env.step(act)
            planner.observe_transition(obs.cylinder - prev.cylinder, obs.finger - prev.finger)
            if frames is not None:
                frames.append(env.render(a.camera, a.width, a.height))
            if obs.collision:
                collided = True; break
            if obs.success:
                break
        if frames is None:
            continue
        tag = "success" if (obs.success and not collided) else ("fail_collision" if collided else "fail_timeout")
        name = out / f"episode_{ep:03d}_{tag}"
        frames += [frames[-1]] * 8
        imageio.mimwrite(str(name) + ".mp4", frames, fps=int(1 / cfg.control.dt), quality=8)
        vf = "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(name) + ".mp4", "-vf", vf, str(name) + ".gif"], check=True)
        if not a.keep_mp4:
            os.remove(str(name) + ".mp4")
        print(f"episode {ep}: {tag}, {t + 1} steps -> {name}.gif ({Path(str(name) + '.gif').stat().st_size / 1e6:.1f} MB)")
    env.close()


if __name__ == "__main__":
    main()
