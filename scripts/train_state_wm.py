"""Train the state world model on trajectory .npz files, split by clip.

Usage: python scripts/train_state_wm.py --data data/processed/video --out models/wm_video_h0.pt --history 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from pushwm.data.trajectories import load_dir, stack_transitions
from pushwm.models.state_wm import evaluate, train_state_wm


def split_by_clip(trajs, val_frac=0.15, test_frac=0.15, seed=0):
    """Split by clip (never by transition): a clip is one continuous push, so
    transitions inside it are correlated and would leak between splits."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trajs))
    n_val = max(1, int(round(val_frac * len(trajs))))
    n_test = max(1, int(round(test_frac * len(trajs))))
    val = [trajs[i] for i in idx[:n_val]]
    test = [trajs[i] for i in idx[n_val:n_val + n_test]]
    train = [trajs[i] for i in idx[n_val + n_test:]]
    return train, val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="trajectory directories")
    p.add_argument("--out", default="outputs/models/state_wm.pt")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--no-rotate", action="store_true")
    p.add_argument("--no-reflect", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=1)
    a = p.parse_args()

    trajs = [t for d in a.data for t in load_dir(d)]
    train, val, test = split_by_clip(trajs, a.val_frac, a.test_frac, a.seed)
    Xtr, Xva, Xte = (stack_transitions(x, a.history) for x in (train, val, test))
    print(f"clips train={len(train)} val={len(val)} test={len(test)} | transitions train={len(Xtr[0])} val={len(Xva[0])} test={len(Xte[0])}")
    model, hist = train_state_wm(*Xtr, val=Xva, n_models=a.n_models, hidden=a.hidden, depth=a.depth,
                                 epochs=a.epochs, rotate=not a.no_rotate, reflect=not a.no_reflect, seed=a.seed)
    metrics = dict(train=evaluate(model, *Xtr), val=evaluate(model, *Xva), test=evaluate(model, *Xte))
    print(json.dumps(metrics, indent=1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), n_models=a.n_models, hidden=a.hidden, depth=a.depth, history=a.history,
                    metrics=metrics, train_history=hist, args=vars(a), val_clips=[t.name for t in val],
                    test_clips=[t.name for t in test], train_clips=[t.name for t in train]), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
