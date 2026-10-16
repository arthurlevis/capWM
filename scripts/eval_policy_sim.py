"""Closed-loop evaluation of an image-based policy (SmolVLA) in our simulator with the
benchmark protocol (100 episodes per layout, seeds 600/700, 80-step limit, failure on
obstacle contact). Writes eval_<layout>_smolvla.json in the benchmark format and
logs for render_all_episodes.py.

    python scripts/eval_policy_sim.py --policy outputs/smolvla/final/pretrained_model
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from pushwm.config import load_config
from pushwm.sim import PushEnv

TASK = "Push the blue cap to the green target without touching the red obstacles."


class SmolVLAAdapter:
    """Wraps a fine-tuned LeRobot SmolVLA policy. Dataset feature names used at
    fine-tuning time must match: observation.images.overhead, observation.state, action.

    LeRobot >= 0.4 keeps tokenisation, device placement and (un)normalisation outside the
    policy, in the pre/post-processor pipelines saved next to the weights, so the raw
    observation is run through the preprocessor and the policy output through the
    postprocessor (verified against lerobot 0.6.1)."""

    def __init__(self, path, device=None):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = SmolVLAPolicy.from_pretrained(path).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def act(self, image, state):
        img = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0)[None]
        batch = {"observation.images.overhead": img,
                 "observation.state": torch.as_tensor(np.asarray(state), dtype=torch.float32)[None],
                 "task": [TASK]}
        batch = self.pre(batch)                       # tokenise task, move to device, normalise state
        a = self.policy.select_action(batch)          # (1, action_dim), normalised; queue of n_action_steps
        a = self.post(a)                              # unnormalise -> metres, on cpu
        return a[0, :2].detach().cpu().numpy().astype(np.float64)


def run(env, adapter, layout, seed, episodes, max_steps, size):
    env.rng = np.random.default_rng(seed)
    results, logs = [], []
    for ep in range(episodes):
        obs = env.reset(opposite_sides=(layout == "opposite"), obstacle_on_line=(layout == "barrier"))
        adapter.reset()
        log = [obs.as_dict()]
        collided = success = False
        for t in range(max_steps):
            a = adapter.act(env.render_overhead(size), obs.finger)
            obs = env.step(np.clip(a, -0.02, 0.02))
            log.append(obs.as_dict())
            if obs.collision:
                collided = True; break
            if obs.success:
                success = True; break
        steps = len(log) - 1
        cl = min((np.linalg.norm(r["cylinder"] - ob[:2]) - ob[2] for r in log for ob in log[0]["obstacles"]), default=None)
        results.append(dict(success=bool(success and not collided), collided=bool(collided), timeout=bool(not success and not collided),
                            steps=steps, final_dist=float(np.linalg.norm(log[-1]["cylinder"] - log[-1]["goal"])),
                            start_dist=float(np.linalg.norm(log[0]["cylinder"] - log[0]["goal"])),
                            path_len=float(sum(np.linalg.norm(log[i + 1]["cylinder"] - log[i]["cylinder"]) for i in range(steps))),
                            min_obstacle_clearance=float(cl) if cl is not None else None))
        logs.append(log)
        if ep % 10 == 0:
            print(f"{layout} ep {ep:3d}: success={results[-1]['success']} collided={collided} steps={steps}")
    succ = float(np.mean([r["success"] for r in results]))
    print(f"{layout}: success rate {succ:.0%} over {episodes} episodes")
    return dict(results=results, success_rate=succ, args=dict(policy="smolvla", layout=layout, seed=seed)), logs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--logs", default="outputs/logs")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--tag", default="smolvla")
    a = p.parse_args()
    cfg = load_config(); env = PushEnv(cfg, seed=0)
    adapter = SmolVLAAdapter(a.policy)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ldir = Path(a.logs); ldir.mkdir(parents=True, exist_ok=True)
    for layout, seed in [("opposite", 600), ("barrier", 700)]:
        res, logs = run(env, adapter, layout, seed, a.episodes, 80, a.size)
        json.dump(res, open(out / f"eval_{layout}_{a.tag}.json", "w"), indent=1)
        np.save(ldir / f"logs_{layout}_{a.tag}.npy", np.array(logs, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    main()
