"""Trajectory files: the one format shared by phone videos and the simulator.

Each trajectory is an ``.npz`` with
    finger   (T, 2)  probe/fingertip xy in box metres
    cylinder (T, 2)  cylinder centre xy in box metres
    valid    (T,)    True where both were tracked
    dt       float   seconds per step (10 Hz after decimation)
    source   str     "video" or "sim"
    name     str     clip or episode name
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trajectory:
    finger: np.ndarray
    cylinder: np.ndarray
    valid: np.ndarray
    dt: float
    source: str
    name: str

    def save(self, path: str | Path):
        np.savez_compressed(path, finger=self.finger, cylinder=self.cylinder, valid=self.valid,
                            dt=self.dt, source=self.source, name=self.name)

    @staticmethod
    def load(path: str | Path) -> "Trajectory":
        z = np.load(path, allow_pickle=False)
        return Trajectory(z["finger"], z["cylinder"], z["valid"].astype(bool), float(z["dt"]),
                          str(z["source"]), str(z["name"]))

    def __len__(self):
        return len(self.finger)


def load_dir(d: str | Path, pattern="*.npz") -> list[Trajectory]:
    return [Trajectory.load(p) for p in sorted(Path(d).glob(pattern))]


def transitions(traj: Trajectory, history: int = 1):
    """Return (vecs, d_cyl) for consecutive valid windows.

    vecs : (N, 2 + 2*history, 2) stack of 2-D vectors, in order
           rel = cylinder - finger at t, d_finger = finger[t+1] - finger[t] (the action),
           then for h = 1..history: d_cyl[t-h], d_finger[t-h] (recent motion, captures sliding).
    d_cyl: (N, 2) cylinder[t+1] - cylinder[t] (the target).
    Rows are kept only when all frames in the window are valid.
    """
    f, c, v = traj.finger, traj.cylinder, traj.valid
    T = len(f)
    if T < history + 2:
        return np.zeros((0, 2 + 2 * history, 2)), np.zeros((0, 2))
    df, dc = np.diff(f, axis=0), np.diff(c, axis=0)       # index t -> t->t+1
    rows_v, rows_y = [], []
    for t in range(history, T - 1):
        if not v[t - history:t + 2].all():
            continue
        blocks = [c[t] - f[t], df[t]]
        for h in range(1, history + 1):
            blocks += [dc[t - h], df[t - h]]
        rows_v.append(np.stack(blocks)); rows_y.append(dc[t])
    if not rows_v:
        return np.zeros((0, 2 + 2 * history, 2)), np.zeros((0, 2))
    return np.stack(rows_v), np.stack(rows_y)


def stack_transitions(trajs: list[Trajectory], history: int = 1):
    parts = [transitions(t, history) for t in trajs]
    vecs = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, 2 + 2 * history, 2))
    d_c = np.concatenate([p[1] for p in parts]) if parts else np.zeros((0, 2))
    return vecs.astype(np.float32), d_c.astype(np.float32)
