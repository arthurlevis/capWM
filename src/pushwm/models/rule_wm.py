"""Hand-written contact model: the textbook alternative to the learned world model.

If the probe centre is within the rigid contact distance of the cap centre
after the move, the cap is pushed just enough to stay outside that distance
(a rigid disc shoved by a rigid disc); otherwise it does not move. No sliding,
no deformation, no fuzziness. Same ``rollout`` interface as StateWorldModel so
the planner is unchanged; the ensemble std is zero.
"""
from __future__ import annotations

import torch


class RuleWorldModel:
    history = 0

    def __init__(self, contact_dist: float, device="cpu"):
        self.contact_dist = float(contact_dist)
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def rollout(self, finger0, cyl0, d_fingers, prev=None, member=None):
        f, c = finger0, cyl0
        fs, cs, ss = [f], [c], []
        for h in range(d_fingers.shape[1]):
            f_new = f + d_fingers[:, h]
            rel = c - f_new
            dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
            push = torch.clamp(self.contact_dist - dist, min=0.0)             # penetration depth
            c = c + push * rel / dist.clamp_min(1e-6)                         # shove cap out along the contact normal
            f = f_new
            fs.append(f); cs.append(c); ss.append(torch.zeros_like(c))
        return torch.stack(fs, 1), torch.stack(cs, 1), torch.stack(ss, 1)
