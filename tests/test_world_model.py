import numpy as np
import torch

from pushwm.models.state_wm import evaluate, train_state_wm


def push_prediction(model, dist_m, step_m):
    vecs = torch.tensor([[[dist_m, 0.0], [step_m, 0.0]]], dtype=torch.float32)
    mean, _ = model.predict(vecs)
    return mean[0, 0].item()


def test_shipped_model_learned_contact(world_model):
    assert push_prediction(world_model, 0.020, 0.010) > 0.007       # in contact: cap moves with the finger
    assert abs(push_prediction(world_model, 0.060, 0.010)) < 0.002  # out of reach: cap stays


def test_rollout_shapes(world_model):
    f0 = torch.zeros(4, 2); c0 = torch.full((4, 2), 0.05); acts = torch.zeros(4, 6, 2)
    fs, cs, ss = world_model.rollout(f0, c0, acts)
    assert fs.shape == (4, 7, 2) and cs.shape == (4, 7, 2) and ss.shape == (4, 6, 2)


def test_training_learns_a_synthetic_contact_rule():
    rng = np.random.default_rng(0); n = 4000
    rel = rng.uniform(-0.06, 0.06, (n, 2)).astype(np.float32); d_f = rng.uniform(-0.015, 0.015, (n, 2)).astype(np.float32)
    touching = np.linalg.norm(rel, axis=1) < 0.022
    d_c = (d_f * touching[:, None]).astype(np.float32)
    vecs = np.stack([rel, d_f], 1)
    model, _ = train_state_wm(vecs, d_c, n_models=2, hidden=128, depth=2, epochs=150, lr=3e-3, log_every=0, device="cpu")
    m = evaluate(model, vecs, d_c)
    assert m["rmse_mm"] < 0.7 * m["rmse_zero_mm"]
