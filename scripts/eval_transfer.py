"""Evaluate a trained world model on another trajectory set (real-to-sim gap).

    python scripts/eval_transfer.py --model outputs/wm_video_h1.pt --data data/processed/sim
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from pushwm.data.trajectories import load_dir, stack_transitions
from pushwm.models.state_wm import StateWorldModel, evaluate


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = StateWorldModel(ck["n_models"], ck["hidden"], ck["depth"], ck.get("history", 1))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--near", type=float, default=0.045, help="finger-cylinder distance (m) defining 'near' transitions")
    a = p.parse_args()
    m, ck = load_model(a.model)
    V, Y = stack_transitions(load_dir(a.data), m.history)
    res = evaluate(m, V, Y)
    rel = np.linalg.norm(V[:, 0], axis=1); near = rel < a.near
    res_near = evaluate(m, V[near], Y[near]); res_far = evaluate(m, V[~near], Y[~near])
    print(json.dumps(dict(model=a.model, data=a.data, n=len(Y), all=res, near=res_near, far=res_far), indent=1))


if __name__ == "__main__":
    main()
