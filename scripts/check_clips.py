"""Per-clip QA: metadata, rim detection on each clip's median background, and
corner drift against a reference calibration (did the phone move?).

    python scripts/check_clips.py --videos data/raw --calib data/calib/box.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pushwm.config import load_config
from pushwm.vision.calib import BoxCalibration, detect_box, median_background, order_corners

EXT = {".mov", ".mp4", ".m4v"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="data/raw")
    p.add_argument("--calib", default="data/calib/box.json")
    p.add_argument("--out", default="data/processed/clip_check.json")
    p.add_argument("--n-frames", type=int, default=15)
    a = p.parse_args()
    cfg = load_config()
    ref = BoxCalibration.load(a.calib)
    rows = {}
    for v in sorted(x for x in Path(a.videos).iterdir() if x.suffix.lower() in EXT):
        cap = cv2.VideoCapture(str(v))
        n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        ok, f = cap.read(); cap.release()
        wh = (f.shape[1], f.shape[0]) if ok else None
        bg = median_background(v, a.n_frames)
        quad, method = detect_box(bg)
        row = dict(frames=n, fps=fps, seconds=round(n / fps, 1) if fps else None, wh=wh, detect=method)
        if quad is not None:
            q = order_corners(quad, cfg.box.size_x, cfg.box.size_y)
            drift = np.linalg.norm(q - ref.corners_px, axis=1)
            row.update(corners=q.round(1).tolist(), max_drift_px=float(drift.max()))
        rows[v.name] = row
        print(f"{v.name}: {n} fr {fps:.0f} fps {row['seconds']}s {wh} {method} "
              f"drift={row.get('max_drift_px', float('nan')):.1f}px")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    d = [r["max_drift_px"] for r in rows.values() if "max_drift_px" in r]
    print(f"{len(rows)} clips; detection ok on {len(d)}; corner drift vs reference: median {np.median(d):.1f}px max {np.max(d):.1f}px")


if __name__ == "__main__":
    main()
