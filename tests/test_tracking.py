import cv2
import numpy as np

from pushwm.vision.extract import decimate, reject_jumps, track_video
from pushwm.vision.track import blue_cylinder_tracker, red_obstacle_mask
from tests.conftest import ROOT


def synthetic_frame():
    img = np.full((600, 400, 3), 235, np.uint8)
    cv2.circle(img, (200, 300), 40, (200, 90, 20), -1)          # blue disc (BGR)
    for c in [(80, 120), (300, 450)]:
        cv2.rectangle(img, (c[0] - 10, c[1] - 10), (c[0] + 10, c[1] + 10), (30, 30, 220), -1)   # red squares
    return img


def test_blue_tracker_finds_disc_centre():
    x, y, r = blue_cylinder_tracker()(synthetic_frame())
    assert abs(x - 200) < 2 and abs(y - 300) < 2 and abs(r - 40) < 3


def test_red_mask_finds_two_obstacles():
    cnts, _ = cv2.findContours(red_obstacle_mask(synthetic_frame()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(cnts) == 2


def test_jump_rejection_flags_only_the_jump():
    pos = np.array([[0, 0], [0.005, 0], [0.2, 0], [0.015, 0]])   # third frame jumps 19.5 cm in one frame
    valid = reject_jumps(pos, np.ones(4, bool), max_speed_mps=1.5, dt=1 / 30)
    assert valid.tolist() == [True, True, False, True]


def test_decimation_averages_frames_into_bins():
    ts = np.arange(9) / 30.0                                   # 9 frames = 3 bins of 0.1 s
    f = np.stack([ts, np.zeros(9)], 1); c = np.stack([np.zeros(9), ts], 1)
    valid = np.ones(9, bool); valid[4] = False                 # one untracked frame in bin 1
    F, C, V = decimate(ts, f, c, valid, 0.1)
    assert len(F) == 3 and V.tolist() == [True, False, True]
    assert np.isclose(F[0, 0], ts[:3].mean())


def test_real_clip_tracks_fingertip_and_cap(calib):
    clip = ROOT / "data/raw/IMG_0527.MOV"
    assert clip.exists(), "download the clips (see README) into data/raw"
    ts, f_px, c_px, valid, fps = track_video(clip, calib, max_frames=150)
    assert abs(fps - 30) < 0.1 and len(ts) == 150
    assert np.isfinite(c_px).all(axis=1).mean() > 0.8         # cap found in nearly every frame
    assert valid[90:150].mean() > 0.5                          # hand is in the box by then
    cap_m = calib.to_box(c_px[np.isfinite(c_px).all(axis=1)])
    assert (cap_m > 0).all() and (cap_m[:, 0] < calib.size_x).all() and (cap_m[:, 1] < calib.size_y).all()
