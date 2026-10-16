"""Video -> trajectory: track fingertip and cylinder per frame, map to box
metres with the calibration, decimate to the control rate."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pushwm.data.trajectories import Trajectory
from pushwm.vision.calib import BoxCalibration
from pushwm.vision.track import FingertipTracker, blue_cylinder_tracker


def track_video(video_path, calib: BoxCalibration, overlay_path=None, max_frames=None, verbose=False):
    """Returns per-frame arrays (t_s, finger_px, cyl_px, valid) at the native frame rate."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tip_tr = FingertipTracker()
    cyl_tr = blue_cylinder_tracker()
    writer = None
    if overlay_path is not None:
        writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W // 2, H // 2))
    ts, fingers, cyls, valid = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if frame.shape[1] != W or frame.shape[0] != H:  # rotation metadata applied late
            W, H = frame.shape[1], frame.shape[0]
        t_ms = int(round(i * 1000.0 / fps))
        tip = tip_tr(frame, t_ms)
        cyl = cyl_tr(frame)
        ts.append(t_ms / 1000.0)
        fingers.append(tip if tip is not None else (np.nan, np.nan))
        cyls.append(cyl[:2] if cyl is not None else (np.nan, np.nan))
        valid.append(tip is not None and cyl is not None)
        if writer is not None:
            vis = frame.copy()
            if tip_tr.landmarks is not None:
                for p in tip_tr.landmarks.astype(int):
                    cv2.circle(vis, tuple(p), 4, (255, 255, 0), -1)
            if tip is not None:
                cv2.circle(vis, (int(tip[0]), int(tip[1])), 14, (255, 0, 255), 3)
            if cyl is not None:
                cv2.circle(vis, (int(cyl[0]), int(cyl[1])), int(cyl[2]), (0, 0, 255), 3)
            cv2.polylines(vis, [calib.corners_px.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            writer.write(cv2.resize(vis, (W // 2, H // 2)))
        i += 1
    cap.release(); tip_tr.close()
    if writer is not None:
        writer.release()
    return np.array(ts), np.array(fingers, float), np.array(cyls, float), np.array(valid, bool), fps


def reject_jumps(pos, valid, max_speed_mps, dt):
    """Mark frames where a tracked point moves implausibly fast (landmark flicker)."""
    v = valid.copy()
    lim = max_speed_mps * dt
    for i in range(1, len(pos)):
        if v[i] and v[i - 1] and np.linalg.norm(pos[i] - pos[i - 1]) > lim:
            v[i] = False
    return v


def decimate(ts, finger_m, cyl_m, valid, dt_out=0.1):
    """Average positions inside consecutive windows of dt_out seconds.
    A window is valid only if every frame in it was tracked."""
    n_bins = int(np.floor((ts[-1] - ts[0]) / dt_out)) + 1 if len(ts) else 0
    bins = np.floor((ts - ts[0]) / dt_out).astype(int)
    F, C, V = np.zeros((n_bins, 2)), np.zeros((n_bins, 2)), np.zeros(n_bins, bool)
    for b in range(n_bins):
        k = bins == b
        if k.any() and valid[k].all():
            F[b], C[b], V[b] = finger_m[k].mean(0), cyl_m[k].mean(0), True
    return F, C, V


def extract_trajectory(video_path, calib: BoxCalibration, dt_out=0.1, overlay_path=None, name=None,
                       max_frames=None, track_cache=None):
    """``track_cache``: optional .npz path; pixel tracks are loaded from it if present
    (MediaPipe is the slow part) and written to it otherwise, so recalibrating is cheap."""
    if track_cache is not None and Path(track_cache).exists():
        z = np.load(track_cache)
        ts, f_px, c_px, valid, fps = z["ts"], z["finger_px"], z["cyl_px"], z["valid"].astype(bool), float(z["fps"])
    else:
        ts, f_px, c_px, valid, fps = track_video(video_path, calib, overlay_path, max_frames)
        if track_cache is not None:
            Path(track_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(track_cache, ts=ts, finger_px=f_px, cyl_px=c_px, valid=valid, fps=fps)
    f_m = np.full_like(f_px, np.nan); c_m = np.full_like(c_px, np.nan)
    f_m[valid] = calib.to_box(f_px[valid]); c_m[valid] = calib.to_box(c_px[valid])
    # A fingertip cannot move faster than ~1.5 m/s; a cylinder pushed by a finger even less.
    valid = reject_jumps(f_m, valid, 1.5, 1.0 / fps)
    valid = reject_jumps(c_m, valid, 1.0, 1.0 / fps)
    F, C, V = decimate(ts, f_m, c_m, valid, dt_out)
    traj = Trajectory(F, C, V, dt_out, "video", name or Path(video_path).stem)
    stats = dict(frames=len(ts), fps=fps, frame_valid=float(valid.mean()) if len(ts) else 0.0,
                 steps=len(F), step_valid=float(V.mean()) if len(F) else 0.0)
    return traj, stats
