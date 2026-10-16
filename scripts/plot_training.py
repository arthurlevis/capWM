"""Training artefacts for a world-model checkpoint: loss curves, predicted-vs-actual
scatter on the test clips, error by finger distance, and open-loop rollout error
against horizon on held-out clips (real actions replayed through the model).

    python scripts/plot_training.py --model models/wm_video_h0.pt --data data/processed/video
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pushwm.data.trajectories import load_dir, stack_transitions
from pushwm.models.state_wm import StateWorldModel


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = StateWorldModel(ck["n_models"], ck["hidden"], ck["depth"], ck.get("history", 1))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def rollout_errors(model, trajs, horizons=(1, 2, 3, 5, 8, 10, 12), window=12):
    """Open-loop: start at a valid window, replay the real finger actions through the
    model, compare predicted cylinder position with the tracked one at each horizon."""
    errs = {h: [] for h in horizons}
    H = max(horizons)
    for t in trajs:
        v = t.valid
        for s in range(model.history, len(t) - H - 1):
            if not v[s - model.history:s + H + 1].all():
                continue
            f0 = torch.tensor(t.finger[s:s + 1], dtype=torch.float32)
            c0 = torch.tensor(t.cylinder[s:s + 1], dtype=torch.float32)
            acts = torch.tensor(np.diff(t.finger[s:s + H + 1], axis=0), dtype=torch.float32)[None]
            prev = [(torch.tensor(t.cylinder[s - k] - t.cylinder[s - k - 1], dtype=torch.float32)[None],
                     torch.tensor(t.finger[s - k] - t.finger[s - k - 1], dtype=torch.float32)[None])
                    for k in range(model.history)]
            _, cs, _ = model.rollout(f0, c0, acts, prev=prev if prev else None)
            for h in horizons:
                errs[h].append(np.linalg.norm(cs[0, h].numpy() - t.cylinder[s + h]))
    return {h: (float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")) for h, e in errs.items()}, \
           {h: len(e) for h, e in errs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/processed/video")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--name", default="world_model", help="output file prefix")
    p.add_argument("--summary-dir", default="docs/results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.name
    sdir = Path(a.summary_dir); sdir.mkdir(parents=True, exist_ok=True)
    model, ck = load(a.model)
    trajs = load_dir(a.data)
    by_name = {t.name: t for t in trajs}
    test = [by_name[n] for n in ck.get("test_clips", []) if n in by_name]
    val = [by_name[n] for n in ck.get("val_clips", []) if n in by_name]

    # 1. curves
    hist = ck["train_history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([h["train"] for h in hist]); ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss")
    ax[1].plot([h["val_rmse_mm"] for h in hist]); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation error, RMSE (mm)")
    pass
    fig.suptitle("World model training"); fig.tight_layout(); fig.savefig(out / f"{tag}_training.png", dpi=130); plt.close(fig)

    # 2. scatter + error by distance on test clips
    V, Y = stack_transitions(test, model.history)
    with torch.no_grad():
        mean, std = model.predict(torch.tensor(V))
    P = mean.numpy(); err = np.linalg.norm(P - Y, axis=1) * 1000
    rel = np.linalg.norm(V[:, 0], axis=1) * 1000
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(Y[:, 0] * 1000, P[:, 0] * 1000, s=4, alpha=.4, label="x"); ax[0].scatter(Y[:, 1] * 1000, P[:, 1] * 1000, s=4, alpha=.4, label="y")
    lim = np.abs(Y).max() * 1000; ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1); ax[0].set_xlabel("actual cap displacement (mm)"); ax[0].set_ylabel("predicted cap displacement (mm)"); ax[0].legend(); ax[0].set_title(f"Predicted vs actual cap displacement (n = {len(Y)} steps)")
    # 3. rollout error vs horizon (val + test)
    ro, n = rollout_errors(model, test + val)
    hs = sorted(ro); ax[1].plot(hs, [ro[h] * 1000 for h in hs], "o-", label="world model")
    ro0 = {}
    for h in hs:  # no-motion baseline at each horizon
        e = []
        for t in test + val:
            for s in range(model.history, len(t) - max(hs) - 1):
                if t.valid[s:s + h + 1].all(): e.append(np.linalg.norm(t.cylinder[s + h] - t.cylinder[s]))
        ro0[h] = float(np.sqrt(np.mean(np.square(e)))) if e else np.nan
    ax[1].plot(hs, [ro0[h] * 1000 for h in hs], "s--", c="gray", label="predict no motion"); ax[1].set_xlabel("horizon (steps of 0.1 s)"); ax[1].set_ylabel("cap position error, RMSE (mm)"); ax[1].legend(); ax[1].set_title(f"Error after several predicted steps (n = {n[hs[0]]} rollouts)")
    fig.tight_layout(); fig.savefig(out / f"{tag}_test.png", dpi=130); plt.close(fig)

    summary = dict(model=a.model, metrics=ck["metrics"], n_test_transitions=int(len(Y)),
                   rollout_rmse_mm={int(h): ro[h] * 1000 for h in hs}, rollout_zero_rmse_mm={int(h): ro0[h] * 1000 for h in hs},
                   train_clips=ck.get("train_clips"), val_clips=ck.get("val_clips"), test_clips=ck.get("test_clips"))
    (sdir / f"{tag}_test_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(test=ck["metrics"]["test"], rollout_rmse_mm={h: round(v * 1000, 2) for h, v in ro.items()},
                          rollout_zero_mm={h: round(v * 1000, 2) for h, v in ro0.items()}), indent=1))
    print("figures:", out / f"{tag}_training.png", out / f"{tag}_test.png", "| summary:", sdir / f"{tag}_test_summary.json")


if __name__ == "__main__":
    main()
