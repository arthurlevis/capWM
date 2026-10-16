"""Per-frame trackers: blue cylinder by colour, index fingertip by MediaPipe."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pushwm.config import ROOT

HAND_MODEL = ROOT / "assets" / "models" / "hand_landmarker.task"
INDEX_TIP = 8


class ColorBlobTracker:
    """Largest blob within an HSV range. Returns (cx, cy, radius_px) or None."""

    def __init__(self, hsv_lo, hsv_hi, min_area=200, max_area=None, hsv_lo2=None, hsv_hi2=None):
        self.lo, self.hi = np.array(hsv_lo), np.array(hsv_hi)
        self.lo2 = np.array(hsv_lo2) if hsv_lo2 is not None else None
        self.hi2 = np.array(hsv_hi2) if hsv_hi2 is not None else None
        self.min_area, self.max_area = min_area, max_area
        self.last = None

    def mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lo, self.hi)
        if self.lo2 is not None:
            m |= cv2.inRange(hsv, self.lo2, self.hi2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return m

    def __call__(self, bgr):
        m = self.mask(bgr)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= self.min_area
                and (self.max_area is None or cv2.contourArea(c) <= self.max_area)]
        if not cnts:
            self.last = None
            return None
        if self.last is not None and len(cnts) > 1:
            # Prefer the blob nearest the previous detection.
            def d(c):
                (x, y), _ = cv2.minEnclosingCircle(c)
                return np.hypot(x - self.last[0], y - self.last[1])
            c = min(cnts, key=d)
        else:
            c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        (x, y), r = cv2.minEnclosingCircle(c)
        self.last = (M["m10"] / M["m00"], M["m01"] / M["m00"], r)
        return self.last


def blue_cylinder_tracker(min_area=200):
    return ColorBlobTracker((95, 110, 70), (128, 255, 255), min_area=min_area)


def red_obstacle_mask(bgr):
    t = ColorBlobTracker((0, 140, 90), (8, 255, 255), hsv_lo2=(170, 140, 90), hsv_hi2=(180, 255, 255))
    return t.mask(bgr)


class FingertipTracker:
    """MediaPipe HandLandmarker in VIDEO mode; returns index-tip (x_px, y_px) or None."""

    def __init__(self, model_path=HAND_MODEL, num_hands=1, det_conf=0.3, track_conf=0.3):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self.mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=det_conf, min_hand_presence_confidence=track_conf,
            min_tracking_confidence=track_conf)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.landmarks = None

    def __call__(self, bgr, t_ms: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.lm.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), int(t_ms))
        if not res.hand_landmarks:
            self.landmarks = None
            return None
        H, W = bgr.shape[:2]
        lms = res.hand_landmarks[0]
        self.landmarks = np.array([[l.x * W, l.y * H] for l in lms])
        tip = lms[INDEX_TIP]
        return (tip.x * W, tip.y * H)

    def close(self):
        self.lm.close()
