import os
from pathlib import Path

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    from pushwm.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def calib():
    from pushwm.vision.calib import BoxCalibration
    return BoxCalibration.load(ROOT / "data/calib/box.json")


@pytest.fixture(scope="session")
def world_model():
    from pushwm.models.state_wm import load_state_wm
    return load_state_wm(ROOT / "models/wm_video_h0.pt", "cpu")


@pytest.fixture
def env(cfg):
    from pushwm.sim import PushEnv
    e = PushEnv(cfg, seed=0)
    yield e
    e.close()
