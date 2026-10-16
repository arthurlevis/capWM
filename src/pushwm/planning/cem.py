"""Cross-entropy-method planner over finger displacement sequences, scored by
rolling the learned world model forward.

Each control step: sample N action sequences of length H, predict the cylinder
path with the world model, score (goal distance, obstacle and wall clearance,
ensemble disagreement, effort), refit a Gaussian to the elites, repeat a few
times, execute only the first action, replan. Replanning every step is what
makes a rough model usable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pushwm.planning.geodesic import GeodesicMap


@dataclass
class PlannerConfig:
    horizon: int = 12
    n_samples: int = 1024
    n_elite: int = 64
    n_iters: int = 4
    max_step: float = 0.02         # metres per control step
    init_std: float = 0.012
    min_std: float = 0.002
    smooth: float = 0.5            # mean/std update smoothing between iterations
    # All terms are in metres so weights are comparable: a 1 cm penetration with
    # weight 1000 costs 0.1, the same as the cylinder being 10 cm from the goal for one step.
    w_goal: float = 1.0            # per-step cylinder-to-goal distance [m]
    w_terminal: float = 5.0        # extra weight on the final cylinder-to-goal distance [m]
    w_approach: float = 0.3        # per-step finger distance to the push point behind the cylinder [m]
    w_obstacle: float = 1000.0     # squared penetration of cylinder/finger into obstacle + clearance [m^2]
    w_wall: float = 1000.0         # squared penetration outside the allowed box area [m^2]
    w_std: float = 2.0             # ensemble disagreement (epistemic uncertainty) [m]
    w_action: float = 5.0          # effort, squared step length [m^2] (tie-breaker)
    w_overlap: float = 1000.0      # finger predicted inside the cylinder (model untrusted there) [m^2]
    clearance: float = 0.020       # planner safety margin cap-edge to obstacle-edge; 15/20/25 mm swept on seeds 500/400, 20 mm halves collisions at equal success
    wall_margin: float = 0.01
    geodesic: bool = True          # distance-to-goal along obstacle-free paths instead of straight-line
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class CEMPlanner:
    def __init__(self, model, sim_cfg, pcfg: PlannerConfig | None = None):
        self.model = model.to(pcfg.device if pcfg else "cpu").eval() if pcfg else model.eval()
        self.history = getattr(model, "history", 0)
        self.cfg = pcfg or PlannerConfig()
        self.sim_cfg = sim_cfg
        self.r_cyl = float(sim_cfg.cylinder.radius)
        self.r_probe = float(sim_cfg.probe.radius)
        self.size = torch.tensor([sim_cfg.box.size_x, sim_cfg.box.size_y], device=self.cfg.device)
        self.gen = torch.Generator(device=self.cfg.device).manual_seed(self.cfg.seed)
        self.prev_mean = None
        self.hist: list[tuple[np.ndarray, np.ndarray]] = []   # recent (d_cyl, d_finger), newest first
        self.last_info = {}
        self.gmap = None
        self._gmap_key = None

    # ------------------------------------------------------------------ state
    def reset(self):
        self.prev_mean = None
        self.hist = []

    def observe_transition(self, d_cyl, d_finger):
        """Call after each executed step so the history inputs match the data."""
        self.hist = [(np.asarray(d_cyl, float), np.asarray(d_finger, float))] + self.hist
        self.hist = self.hist[: self.history]

    # ------------------------------------------------------------------ cost
    def _ensure_map(self, goal, obstacles):
        key = (tuple(np.round(np.asarray(goal, float), 4)), tuple(np.round(np.asarray(obstacles, float).ravel(), 4)))
        if self.cfg.geodesic and key != self._gmap_key:
            self.gmap = GeodesicMap(self.size.cpu().numpy(), np.asarray(goal, float), obstacles,
                                    inflate=self.r_cyl + self.cfg.clearance, wall_margin=self.r_cyl + self.cfg.wall_margin,
                                    device=self.cfg.device)
            self._gmap_key = key

    def cost(self, fingers, cyls, stds, actions, goal, obstacles):
        """fingers/cyls: (B, H+1, 2); stds: (B, H, 2); actions: (B, H, 2)."""
        c = self.cfg
        if self.gmap is not None:
            d_goal = self.gmap(cyls[:, 1:])                                       # (B, H) distance-to-go around obstacles
            g_dir = self.gmap.descent_dir(cyls[:, 1:])
        else:
            d_goal = torch.linalg.norm(cyls[:, 1:] - goal, dim=-1)
            g_dir = goal - cyls[:, 1:]
            g_dir = g_dir / torch.linalg.norm(g_dir, dim=-1, keepdim=True).clamp_min(1e-6)
        cost = c.w_goal * d_goal.sum(-1) + c.w_terminal * d_goal[:, -1]
        # Shaping: the ideal push point sits behind the cylinder, opposite the way to go.
        push_pt = cyls[:, 1:] - g_dir * (self.r_cyl + self.r_probe)
        cost = cost + c.w_approach * torch.linalg.norm(fingers[:, 1:] - push_pt, dim=-1).sum(-1)
        if obstacles is not None and len(obstacles):
            ob = obstacles[:, :2][None, None]                                    # (1,1,K,2)
            r_ob = obstacles[:, 2][None, None]                                   # (1,1,K)
            d_c = torch.linalg.norm(cyls[:, 1:, None] - ob, dim=-1)              # (B,H,K)
            d_f = torch.linalg.norm(fingers[:, 1:, None] - ob, dim=-1)
            pen_c = torch.relu(r_ob + self.r_cyl + c.clearance - d_c)
            pen_f = torch.relu(r_ob + self.r_probe + c.clearance - d_f)
            cost = cost + c.w_obstacle * ((pen_c ** 2).sum((-1, -2)) + (pen_f ** 2).sum((-1, -2)))
        lo_c = self.r_cyl + c.wall_margin
        lo_f = self.r_probe + c.wall_margin
        pen_wall = (torch.relu(lo_c - cyls[:, 1:]) + torch.relu(cyls[:, 1:] - (self.size - lo_c))) ** 2
        pen_wall_f = (torch.relu(lo_f - fingers[:, 1:]) + torch.relu(fingers[:, 1:] - (self.size - lo_f))) ** 2
        cost = cost + c.w_wall * (pen_wall.sum((-1, -2)) + pen_wall_f.sum((-1, -2)))
        overlap = torch.relu(self.r_cyl + self.r_probe - 0.004 - torch.linalg.norm(cyls[:, 1:] - fingers[:, 1:], dim=-1))
        cost = cost + c.w_overlap * (overlap ** 2).sum(-1)
        cost = cost + c.w_std * stds.sum((-1, -2))
        cost = cost + c.w_action * (actions ** 2).sum((-1, -2))
        return cost

    # ------------------------------------------------------------------ plan
    @torch.no_grad()
    def plan(self, finger, cylinder, goal, obstacles):
        c = self.cfg; dev = c.device
        H, N = c.horizon, c.n_samples
        f0 = torch.as_tensor(finger, dtype=torch.float32, device=dev)[None].expand(N, 2)
        c0 = torch.as_tensor(cylinder, dtype=torch.float32, device=dev)[None].expand(N, 2)
        g = torch.as_tensor(goal, dtype=torch.float32, device=dev)
        ob = torch.as_tensor(np.asarray(obstacles, float).reshape(-1, 3), dtype=torch.float32, device=dev)
        self._ensure_map(goal, obstacles)
        prev = [(torch.as_tensor(dc, dtype=torch.float32, device=dev)[None].expand(N, 2),
                 torch.as_tensor(df, dtype=torch.float32, device=dev)[None].expand(N, 2)) for dc, df in self.hist]

        if self.prev_mean is not None:   # warm start: shift last plan by one step
            mean = torch.cat([self.prev_mean[1:], self.prev_mean[-1:]], 0)
        else:
            mean = torch.zeros(H, 2, device=dev)
        std = torch.full((H, 2), c.init_std, device=dev)

        best_cost, best_seq, best_roll = None, None, None
        for it in range(c.n_iters):
            acts = mean[None] + std[None] * torch.randn(N, H, 2, device=dev, generator=self.gen)
            acts[0] = mean                                           # keep the current mean as a candidate
            n = torch.linalg.norm(acts, dim=-1, keepdim=True)
            acts = torch.where(n > c.max_step, acts * (c.max_step / n), acts)
            fs, cs, ss = self.model.rollout(f0, c0, acts, prev=prev)
            cost = self.cost(fs, cs, ss, acts, g, ob)
            elite_idx = torch.topk(-cost, c.n_elite).indices
            elite = acts[elite_idx]
            mean = c.smooth * mean + (1 - c.smooth) * elite.mean(0)
            std = c.smooth * std + (1 - c.smooth) * elite.std(0)
            std = std.clamp_min(c.min_std)
            i_best = int(torch.argmin(cost))
            if best_cost is None or cost[i_best] < best_cost:
                best_cost, best_seq = float(cost[i_best]), acts[i_best].clone()
                best_roll = (fs[i_best].cpu().numpy(), cs[i_best].cpu().numpy())
        self.prev_mean = best_seq
        self.last_info = dict(cost=best_cost, plan=best_seq.cpu().numpy(), pred_finger=best_roll[0], pred_cyl=best_roll[1])
        return best_seq[0].cpu().numpy()
