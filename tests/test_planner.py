import numpy as np
import torch

from pushwm.models.rule_wm import RuleWorldModel
from pushwm.planning.cem import CEMPlanner, PlannerConfig
from pushwm.planning.geodesic import GeodesicMap


def test_geodesic_distance_goes_around_a_barrier():
    goal = np.array([0.14, 0.30]); start = np.array([0.14, 0.10])
    barrier = np.array([[0.10, 0.20, 0.0085], [0.14, 0.20, 0.0085], [0.18, 0.20, 0.0085]])
    free = GeodesicMap((0.28, 0.37), goal, np.zeros((0, 3)), inflate=0.03, wall_margin=0.02)
    blocked = GeodesicMap((0.28, 0.37), goal, barrier, inflate=0.03, wall_margin=0.02)
    s = torch.tensor(start[None], dtype=torch.float32)
    assert abs(free(s).item() - 0.20) < 0.01
    assert blocked(s).item() > free(s).item() + 0.03


def test_rule_planner_reaches_goal_without_obstacles(env, cfg):
    pcfg = PlannerConfig(n_samples=256, n_iters=2, seed=0, device="cpu")
    planner = CEMPlanner(RuleWorldModel(cfg.cylinder.radius + cfg.probe.radius), cfg, pcfg)
    obs = env.reset(start=[0.14, 0.12], goal=[0.14, 0.26], n_obstacles=0)
    planner.reset()
    for _ in range(80):
        a = planner.plan(obs.finger, obs.cylinder, obs.goal, obs.obstacles)
        prev = obs; obs = env.step(a)
        planner.observe_transition(obs.cylinder - prev.cylinder, obs.finger - prev.finger)
        if obs.success:
            break
    assert obs.success
