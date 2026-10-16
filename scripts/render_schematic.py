"""Render episodes from saved logs as a state-space animation: box, obstacles,
goal, cylinder and finger trails, and the planner's predicted cylinder path at
each step. This is the view the planner actually reasons in.

    python scripts/render_schematic.py --logs outputs/logs_hard_video_h0.npy --episodes 0 1 2 --out docs/media/schematic_hard.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

from pushwm.config import load_config


def draw_frame(ax, cfg, log, t, title):
    ax.clear()
    sx, sy = cfg.box.size_x, cfg.box.size_y
    ax.add_patch(Rectangle((0, 0), sx, sy, fill=False, lw=3, color="#222"))
    rec = log[t]
    for ob in rec["obstacles"]:
        ax.add_patch(Circle(ob[:2], ob[2], color="#d33", alpha=.9))
    ax.add_patch(Circle(rec["goal"], cfg.task.goal_tol, color="#2a2", alpha=.35))
    cyl = np.array([r["cylinder"] for r in log[: t + 1]]); fin = np.array([r["finger"] for r in log[: t + 1]])
    ax.plot(cyl[:, 0], cyl[:, 1], "-", color="#1f77b4", lw=2, alpha=.7, label="cylinder path")
    ax.plot(fin[:, 0], fin[:, 1], "-", color="#ff7f0e", lw=1.2, alpha=.7, label="finger path")
    if "pred_cyl" in rec:
        pc = np.asarray(rec["pred_cyl"]); pf = np.asarray(rec["pred_finger"])
        ax.plot(pc[:, 0], pc[:, 1], "o--", color="#1f77b4", ms=3, lw=1, alpha=.9, label="world-model plan (cylinder)")
        ax.plot(pf[:, 0], pf[:, 1], "--", color="#ff7f0e", lw=1, alpha=.9, label="plan (finger)")
    ax.add_patch(Circle(rec["cylinder"], cfg.cylinder.radius, color="#1f77b4"))
    ax.add_patch(Circle(rec["finger"], cfg.probe.radius, color="#ff7f0e"))
    ax.set_xlim(-0.02, sx + 0.02); ax.set_ylim(-0.02, sy + 0.02); ax.set_aspect("equal"); ax.axis("off")
    status = "SUCCESS" if rec["success"] else ("COLLISION - FAIL" if rec.get("collision") else "")
    ax.set_title(f"{title}\nstep {t}  {status}", fontsize=9)
    if t == 0:
        ax.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, 0), ncol=2, frameon=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", required=True, help="one or more logs npy (rendered side by side)")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--episodes", nargs="+", type=int, default=[0])
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    all_logs = [np.load(l, allow_pickle=True) for l in a.logs]
    labels = a.labels or [Path(l).stem.replace("logs_", "") for l in a.logs]
    frames = []
    for ep in a.episodes:
        logs = [L[ep] for L in all_logs]
        T = max(len(l) for l in logs)
        fig, axes = plt.subplots(1, len(logs), figsize=(4.2 * len(logs), 4.8))
        fig.subplots_adjust(wspace=0.35, left=0.03, right=0.97, top=0.92, bottom=0.08)
        axes = np.atleast_1d(axes)
        for t in range(T):
            for ax, log, lab in zip(axes, logs, labels):
                draw_frame(ax, cfg, log, min(t, len(log) - 1), f"ep {ep} {lab}")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        for _ in range(a.fps):   # hold the final frame
            frames.append(frames[-1])
        plt.close(fig)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        imageio.mimwrite(out, frames, duration=1000 / a.fps, loop=0)
    else:
        imageio.mimwrite(out, frames, fps=a.fps, quality=7)
    print(out, f"{out.stat().st_size/1e6:.1f} MB", len(frames), "frames")


if __name__ == "__main__":
    main()
