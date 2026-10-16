"""One-time box calibration for a recording session.

    python scripts/calibrate_box.py --video data/raw/clip01.mp4
        -> writes data/calib/background.jpg and an auto-detected guess overlay.
    python scripts/calibrate_box.py --video data/raw/clip01.mp4 --corners x0,y0 x1,y1 x2,y2 x3,y3
        -> saves data/calib/box.json from the given inner-floor corners (pixels, any
           order around the rectangle; the short side is identified automatically).
    python scripts/calibrate_box.py --video ... --click
        -> opens a matplotlib window; click the four inner-floor corners.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pushwm.config import load_config
from pushwm.vision.calib import BoxCalibration, detect_box, draw_calibration, median_background, order_corners


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--corners", nargs=4, default=None, help="four 'x,y' pixel pairs")
    p.add_argument("--click", action="store_true")
    p.add_argument("--out", default="data/calib/box.json")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    out_dir = Path(a.out).parent; out_dir.mkdir(parents=True, exist_ok=True)

    bg = median_background(a.video)
    cv2.imwrite(str(out_dir / "background.jpg"), bg)
    H, W = bg.shape[:2]
    quad, method = detect_box(bg)
    if quad is not None:
        guess = order_corners(quad, cfg.box.size_x, cfg.box.size_y)
        print(f"auto-detected corners ({method}):", guess.astype(int).tolist())
    else:
        guess = None
        print("auto-detection failed")

    corners = None
    if a.corners:
        corners = np.array([[float(v) for v in c.split(",")] for c in a.corners])
    elif a.click:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 11))
        ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        if guess is not None:
            ax.plot(*np.vstack([guess, guess[:1]]).T, "y--", lw=1, label="auto guess")
        ax.set_title("Click the 4 INNER floor corners, going around the rectangle")
        pts = plt.ginput(4, timeout=0)
        plt.close(fig)
        corners = np.array(pts)
    elif guess is not None:
        corners = guess
        print("using the detected corners; check data/calib/calibration_check.jpg and re-run with --corners if it is off")
    else:
        raise SystemExit("rim detection failed: re-run with --click or --corners x,y x,y x,y x,y")

    corners = order_corners(corners, cfg.box.size_x, cfg.box.size_y)
    calib = BoxCalibration(corners, cfg.box.size_x, cfg.box.size_y, (W, H))
    calib.save(a.out)
    cv2.imwrite(str(out_dir / "calibration_check.jpg"), draw_calibration(bg, calib))
    e = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    print(f"saved {a.out}; edges px {np.round(e).astype(int).tolist()}; ~{calib.px_per_m():.0f} px/m;"
          f" check {out_dir / 'calibration_check.jpg'}")


if __name__ == "__main__":
    main()
