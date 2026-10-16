"""Render every logged episode as its own schematic GIF, named by index and outcome.

    python scripts/render_all_episodes.py --logs outputs/logs_random_video_h0.npy --out docs/media/random
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pushwm.config import load_config
from render_schematic import draw_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="world-model planner")
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args()
    cfg = load_config()
    logs = np.load(a.logs, allow_pickle=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ep, log in enumerate(logs):
        collided = any(bool(r.get("collision", False)) for r in log)
        ok = bool(log[-1]["success"]) and not collided
        tag = "success" if ok else ("fail_collision" if collided else "fail_timeout")
        fig, ax = plt.subplots(figsize=(3.0, 3.9))
        frames = []
        for t in range(len(log)):
            draw_frame(ax, cfg, log, t, f"ep {ep} {a.label}")
            fig.canvas.draw(); frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        frames += [frames[-1]] * a.fps
        plt.close(fig)
        name = out / f"episode_{ep:03d}_{tag}.gif"
        imageio.mimwrite(name, frames, duration=1000 / a.fps, loop=0)
    n_ok = sum(bool(l[-1]["success"]) and not any(bool(r.get("collision", False)) for r in l) for l in logs)
    print(f"{len(logs)} episodes -> {out} ({n_ok} success, {len(logs)-n_ok} fail)")


if __name__ == "__main__":
    main()
