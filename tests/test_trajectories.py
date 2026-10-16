import numpy as np
import torch

from pushwm.data.trajectories import Trajectory, transitions
from pushwm.models.state_wm import augment


def test_transitions_skip_invalid_frames():
    f = np.zeros((6, 2)); c = np.zeros((6, 2)); c[:, 0] = np.arange(6) * 0.01
    valid = np.array([True, True, False, True, True, True])
    V, Y = transitions(Trajectory(f, c, valid, 0.1, "test", "t"), history=0)
    assert len(Y) == 3 and np.allclose(Y[:, 0], 0.01)


def test_augmentation_preserves_lengths_and_relations():
    vecs = torch.tensor([[[0.02, 0.0], [0.01, 0.0]]]).repeat(50, 1, 1); d_c = torch.tensor([[0.01, 0.0]]).repeat(50, 1)
    v2, y2 = augment(vecs, d_c, rotate=True, reflect=True)
    assert torch.allclose(v2.norm(dim=-1), vecs.norm(dim=-1), atol=1e-6)
    # the cap moved along the finger's direction; after augmentation it still does
    cos = (v2[:, 1] * y2).sum(-1) / (v2[:, 1].norm(dim=-1) * y2.norm(dim=-1))
    assert torch.allclose(cos, torch.ones(50), atol=1e-5)
