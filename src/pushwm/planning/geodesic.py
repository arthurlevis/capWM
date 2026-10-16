"""Geodesic distance-to-goal on an obstacle-inflated grid.

Straight-line distance makes a sampling planner short-sighted: detouring around
a barrier looks worse before it looks better. A Dijkstra distance map from the
goal, with obstacles and walls inflated by the cylinder radius plus clearance,
gives a cost that already knows the way around. The world model still decides
what the finger must do to move the cylinder along it.
"""
from __future__ import annotations

import heapq

import numpy as np
import torch


class GeodesicMap:
    def __init__(self, size_xy, goal, obstacles, inflate, wall_margin, cell=0.005, device="cpu"):
        self.cell = cell
        self.size = np.asarray(size_xy, float)
        nx, ny = int(np.ceil(self.size[0] / cell)) + 1, int(np.ceil(self.size[1] / cell)) + 1
        xs, ys = np.arange(nx) * cell, np.arange(ny) * cell
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        free = (X >= wall_margin) & (X <= self.size[0] - wall_margin) & (Y >= wall_margin) & (Y <= self.size[1] - wall_margin)
        for ox, oy, r in np.asarray(obstacles, float).reshape(-1, 3):
            free &= (X - ox) ** 2 + (Y - oy) ** 2 > (r + inflate) ** 2
        dist = np.full((nx, ny), np.inf)
        gi, gj = int(round(goal[0] / cell)), int(round(goal[1] / cell))
        gi, gj = np.clip(gi, 0, nx - 1), np.clip(gj, 0, ny - 1)
        dist[gi, gj] = 0.0
        heap = [(0.0, gi, gj)]
        nbrs = [(di, dj, np.hypot(di, dj) * cell) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, w in nbrs:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b] and d + w < dist[a, b]:
                    dist[a, b] = d + w
                    heapq.heappush(heap, (d + w, a, b))
        # Blocked cells: straight-line distance plus a penalty so the gradient still points out.
        blocked = ~np.isfinite(dist)
        euclid = np.hypot(X - goal[0], Y - goal[1])
        dist[blocked] = euclid[blocked] + 0.3
        self.free = free
        self.dist = torch.as_tensor(dist, dtype=torch.float32, device=device)
        self.nx, self.ny = nx, ny
        self.device = device

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear lookup. xy: (..., 2) metres -> (...) distance-to-go in metres."""
        g = xy / self.cell
        gx = g[..., 0].clamp(0, self.nx - 1 - 1e-4); gy = g[..., 1].clamp(0, self.ny - 1 - 1e-4)
        i0, j0 = gx.floor().long(), gy.floor().long()
        fx, fy = gx - i0, gy - j0
        d = self.dist
        return ((1 - fx) * (1 - fy) * d[i0, j0] + fx * (1 - fy) * d[i0 + 1, j0]
                + (1 - fx) * fy * d[i0, j0 + 1] + fx * fy * d[i0 + 1, j0 + 1])

    def descent_dir(self, xy: torch.Tensor, eps=0.006) -> torch.Tensor:
        """Unit direction of steepest descent of the distance map at xy (..., 2)."""
        ex = torch.tensor([eps, 0.0], device=xy.device); ey = torch.tensor([0.0, eps], device=xy.device)
        gx = (self(xy + ex) - self(xy - ex)) / (2 * eps)
        gy = (self(xy + ey) - self(xy - ey)) / (2 * eps)
        g = torch.stack([-gx, -gy], -1)
        return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
