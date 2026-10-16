"""Model insight figures: the learned contact function (predicted cylinder motion
vs finger distance and step size, with ensemble uncertainty) for the
no-history and history models, and the contact-distance histogram from the data.

    python scripts/plot_model_insights.py --out docs/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pushwm.config import load_config
from pushwm.data.trajectories import load_dir, stack_transitions
from pushwm.models.state_wm import StateWorldModel


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = StateWorldModel(ck["n_models"], ck["hidden"], ck["depth"], ck.get("history", 1)); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def contact_curve(m, dists, step, moving_history):
    out, sd = [], []
    for d in dists:
        blocks = [torch.tensor([[float(d), 0.0]], dtype=torch.float32), torch.tensor([[float(step), 0.0]], dtype=torch.float32)]
        for _ in range(m.history):
            blocks += [torch.tensor([[float(step) if moving_history else 0.0, 0.0]], dtype=torch.float32)] * 2
        mean, std = m.predict(torch.stack(blocks, dim=-2))
        out.append(mean[0, 0].item()); sd.append(std[0, 0].item())
    return np.array(out), np.array(sd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h0", default="outputs/wm_video_h0.pt")
    p.add_argument("--h1", default="outputs/wm_video_h1.pt")
    p.add_argument("--data", default="data/processed/video")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args()
    cfg = load_config(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dists = np.linspace(0.010, 0.080, 71)
    m0, m1 = load(a.h0), load(a.h1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for step, c in [(0.005, "#1f77b4"), (0.010, "#ff7f0e"), (0.020, "#2ca02c")]:
        mu, sd = contact_curve(m0, dists, step, False)
        ax[0].plot(dists * 1000, mu * 1000, c=c, label=f"finger step {step*1000:.0f} mm"); ax[0].fill_between(dists * 1000, (mu - sd) * 1000, (mu + sd) * 1000, color=c, alpha=.2)
        ax[0].axhline(step * 1000, c=c, ls=":", lw=.8)
    r_contact = cfg.cylinder.radius + cfg.probe.radius
    ax[0].axvline(r_contact * 1000, c="k", ls="--", lw=1.2, label="contact distance")
    ax[0].set_xlabel("finger–cap distance (mm)"); ax[0].set_ylabel("predicted cap move along the push (mm)"); ax[0].set_title("Learned contact function"); ax[0].legend(fontsize=8)
    V, Y = stack_transitions(load_dir(a.data), 0)
    rel = np.linalg.norm(V[:, 0], axis=1) * 1000; mv = np.linalg.norm(Y, axis=1) > 0.003
    ax[1].hist(rel[mv], bins=np.arange(0, 60, 1.5), color="#1f77b4", alpha=.8, label="cap moving"); ax[1].hist(rel[~mv & (rel < 60)], bins=np.arange(0, 60, 1.5), color="gray", alpha=.5, label="cap still")
    ax[1].axvline(r_contact * 1000, c="k", ls="--", lw=1.2, label="contact distance")
    ax[1].set_xlabel("finger–cap distance (mm)"); ax[1].set_ylabel("transitions"); ax[1].set_title("Cap motion based on finger–cap distance"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "contact_function.png", dpi=130); plt.close(fig)
    print("saved", out / "contact_function.png")


if __name__ == "__main__":
    main()
