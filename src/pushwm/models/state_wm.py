"""State-space world model: an ensemble of small MLPs predicting how the
cylinder moves when the finger moves.

Input  : rel = cylinder - finger, d_finger (the action), and ``history`` steps of
         recent (d_cyl, d_finger) so sliding after a fast push is predictable   [metres]
Output : d_cyl (2)                                                                [metres]

Design choices
* Relative coordinates make the model translation-invariant, so a few dozen
  clips cover the whole box.
* Optional rotation/reflection augmentation encodes that finger-on-cylinder
  pushing is isotropic; this is a physics prior, not a trick, and it is ablated.
* An ensemble gives epistemic uncertainty: disagreement is large for finger
  motions never seen in the videos, and the planner can penalise it.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in=4, d_out=2, hidden=256, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StateWorldModel(nn.Module):
    """Ensemble of MLPs over a stack of 2-D vectors (see data.trajectories.transitions)."""

    def __init__(self, n_models=5, hidden=256, depth=3, history=1, scale=0.02):
        super().__init__()
        self.history = history
        self.n_vecs = 2 + 2 * history
        self.members = nn.ModuleList([MLP(2 * self.n_vecs, 2, hidden, depth) for _ in range(n_models)])
        # Per-vector scaling: relative position ~0.1 m, displacements ~0.02 m.
        sc = torch.tensor([0.1] + [scale] * (self.n_vecs - 1)).repeat_interleave(2)
        self.register_buffer("in_scale", sc)
        self.register_buffer("out_scale", torch.tensor([scale, scale]))

    def forward(self, vecs):
        """vecs: (..., n_vecs, 2) -> d_cyl per member: (n_models, ..., 2)."""
        x = vecs.flatten(-2) / self.in_scale
        outs = torch.stack([m(x) for m in self.members], dim=0)
        return outs * self.out_scale

    def predict(self, vecs):
        outs = self.forward(vecs)
        return outs.mean(0), outs.std(0)

    @torch.no_grad()
    def rollout(self, finger0, cyl0, d_fingers, prev=None, member=None):
        """Roll an action sequence forward.

        finger0, cyl0 : (B, 2); d_fingers : (B, H, 2)
        prev : optional list of (d_cyl, d_finger) pairs for the ``history`` most recent
               steps, most recent first, each (B, 2). Zeros if omitted.
        Returns fingers (B, H+1, 2), cyls (B, H+1, 2), std (B, H, 2).
        """
        B = finger0.shape[0]
        z = torch.zeros(B, 2, device=finger0.device, dtype=finger0.dtype)
        hist = list(prev) if prev is not None else [(z, z)] * self.history
        hist = (hist + [(z, z)] * self.history)[: self.history]
        f, c = finger0, cyl0
        fs, cs, ss = [f], [c], []
        for h in range(d_fingers.shape[1]):
            a = d_fingers[:, h]
            blocks = [c - f, a]
            for dc_prev, df_prev in hist:
                blocks += [dc_prev, df_prev]
            outs = self.forward(torch.stack(blocks, dim=-2))
            d_c = outs[member] if member is not None else outs.mean(0)
            ss.append(outs.std(0))
            f = f + a
            c = c + d_c
            fs.append(f); cs.append(c)
            if self.history:
                hist = [(d_c, a)] + hist[:-1]
        return torch.stack(fs, 1), torch.stack(cs, 1), torch.stack(ss, 1)


def rot2d(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def augment(vecs, d_c, rotate=True, reflect=True):
    """Apply one random rotation (and reflection) per row to every 2-D vector."""
    B = vecs.shape[0]
    if rotate:
        R = rot2d(torch.rand(B, device=vecs.device) * 2 * math.pi)       # (B, 2, 2)
        vecs = torch.einsum("bij,bkj->bki", R, vecs)
        d_c = (R @ d_c.unsqueeze(-1)).squeeze(-1)
    if reflect:
        flip = (torch.rand(B, device=vecs.device) < 0.5).float() * -2 + 1  # ±1 on y
        vecs = vecs * torch.stack([torch.ones_like(flip), flip], -1)[:, None, :]
        d_c = d_c * torch.stack([torch.ones_like(flip), flip], -1)
    return vecs, d_c


def train_state_wm(vecs, d_c, val=None, n_models=5, hidden=256, depth=3, epochs=300,
                   batch=512, lr=1e-3, weight_decay=1e-5, rotate=True, reflect=True,
                   device=None, log_every=50, seed=0):
    """Train the ensemble with bootstrap resampling per member. Returns (model, history)."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    history = (vecs.shape[1] - 2) // 2
    model = StateWorldModel(n_models, hidden, depth, history).to(device)
    X = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in (vecs, d_c)]
    N = X[0].shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    boot = [torch.randint(0, N, (N,), generator=g) for _ in range(n_models)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = [b[torch.randperm(N, generator=g)] for b in boot]
        tot, nb = 0.0, 0
        for i in range(0, N, batch):
            loss = 0.0
            for k, m in enumerate(model.members):
                idx = perm[k][i:i + batch]
                v, y = augment(X[0][idx], X[1][idx], rotate, reflect)
                pred = m(v.flatten(-2) / model.in_scale) * model.out_scale
                loss = loss + ((pred - y) ** 2).sum(-1).mean() / model.out_scale[0] ** 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        rec = dict(epoch=ep, train=tot / max(nb, 1))
        if val is not None:
            rec["val_rmse_mm"] = evaluate(model, *val)["rmse_mm"]
        hist.append(rec)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d} train={rec['train']:.4f}" + (f" val_rmse={rec['val_rmse_mm']:.2f}mm" if val is not None else ""))
    model.eval()
    return model, hist


@torch.no_grad()
def evaluate(model, vecs, d_c):
    device = next(model.parameters()).device
    v, y = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (vecs, d_c)]
    mean, std = model.predict(v)
    err = mean - y
    rmse = float(torch.sqrt((err ** 2).sum(-1).mean())) * 1000
    # Baselines: predict zero motion, and predict "cylinder copies finger".
    rmse_zero = float(torch.sqrt((y ** 2).sum(-1).mean())) * 1000
    moving = (y.norm(dim=-1) > 0.002)
    rmse_moving = float(torch.sqrt((err[moving] ** 2).sum(-1).mean())) * 1000 if moving.any() else float("nan")
    return dict(rmse_mm=rmse, rmse_zero_mm=rmse_zero, rmse_moving_mm=rmse_moving,
                frac_moving=float(moving.float().mean()), mean_std_mm=float(std.mean()) * 1000)


def load_state_wm(path, device="cpu"):
    """Load a checkpoint written by scripts/train_state_wm.py."""
    ck = torch.load(path, map_location=device, weights_only=False)
    m = StateWorldModel(ck["n_models"], ck["hidden"], ck["depth"], ck.get("history", 1))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()
