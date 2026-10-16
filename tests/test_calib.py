import cv2
import numpy as np

from pushwm.vision.calib import BoxCalibration, fit_rim_lines, median_background, order_corners
from tests.conftest import ROOT


def test_pixel_metre_round_trip(calib):
    pts = np.array([[0.05, 0.05], [0.20, 0.30], [0.14, 0.185]])
    back = calib.to_box(calib.to_px(pts))
    assert np.allclose(back, pts, atol=1e-6)


def test_corner_order_puts_short_side_first():
    quad = np.array([[100, 100], [100, 900], [700, 900], [700, 100]], float)   # 600 wide (short), 800 tall (long)
    q = order_corners(quad, size_x=0.28, size_y=0.37)
    assert np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[2] - q[1])


def test_rim_detection_on_a_clip_matches_stored_calibration(calib):
    clip = ROOT / "data/raw/IMG_0527.MOV"
    assert clip.exists(), "download the clips (see README) into data/raw"
    quad = fit_rim_lines(median_background(clip, 15))
    assert quad is not None
    q = order_corners(quad, calib.size_x, calib.size_y)
    # the stored calibration is the median over all clips; single-clip detections scatter by up to ~30 px
    assert np.linalg.norm(q - calib.corners_px, axis=1).max() < 40
