"""Per-clip rim detection on each clip's median background, a QA contact sheet,
and per-clip calibrations (falling back to the session median for failures).

    python scripts/calibrate_all.py --videos data/raw --out data/calib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pushwm.config import load_config
from pushwm.vision.calib import BoxCalibration, detect_box, draw_calibration, median_background, order_corners

EXT = {".mov", ".mp4", ".m4v"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="data/raw")
    p.add_argument("--out", default="data/calib")
    p.add_argument("--n-frames", type=int, default=15)
    p.add_argument("--max-size-dev", type=float, default=0.04, help="relative edge-length deviation from the session median beyond which a detection is rejected")
    a = p.parse_args()
    cfg = load_config()
    sx, sy = cfg.box.size_x + 2 * cfg.box.wall_thickness, cfg.box.size_y + 2 * cfg.box.wall_thickness
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vids = sorted(x for x in Path(a.videos).iterdir() if x.suffix.lower() in EXT)
    bgs, dets = {}, {}
    for v in vids:
        bg = median_background(v, a.n_frames); bgs[v.stem] = bg
        quad, method = detect_box(bg)
        dets[v.stem] = order_corners(quad, sx, sy) if quad is not None else None
    def edges(q):
        return np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])
    good = {k: q for k, q in dets.items() if q is not None}
    med = np.median(np.stack(list(good.values())), axis=0)
    med_edges = np.median(np.stack([edges(q) for q in good.values()]), axis=0)
    # The phone can shift slightly between takes, so a clip keeps its own
    # detection whenever the rectangle has the right size; a failed or
    # implausible detection borrows the nearest-in-time plausible one.
    keys = list(dets)
    ok = {k: (dets[k] is not None and np.all(np.abs(edges(dets[k]) / med_edges - 1) < a.max_size_dev)) for k in keys}
    final, status = {}, {}
    for i, k in enumerate(keys):
        if ok[k]:
            dev = float(np.linalg.norm(dets[k] - med, axis=1).max())
            final[k], status[k] = dets[k], f"own(dev {dev:.0f}px)"
        else:
            j = min((j for j in range(len(keys)) if ok[keys[j]]), key=lambda j: abs(j - i))
            final[k], status[k] = dets[keys[j]], f"from {keys[j]}"
    H, W = next(iter(bgs.values())).shape[:2]
    tiles = []
    for k in sorted(final):
        cal = BoxCalibration(final[k], sx, sy, (W, H))
        vis = draw_calibration(bgs[k], cal, grid=0.1)
        cv2.putText(vis, f"{k} {status[k]}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        tiles.append(cv2.resize(vis[: int(H * 0.62)], (300, int(H * 0.62 * 300 / W))))
    cols = 8
    rows = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), axis=1)
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out / "per_clip_sheet.jpg"), np.concatenate(rows, axis=0))
    json.dump({k: dict(corners_px=final[k].tolist(), status=status[k], size_x=sx, size_y=sy, image_wh=[W, H])
               for k in final}, open(out / "per_clip.json", "w"), indent=1)
    BoxCalibration(med, sx, sy, (W, H)).save(out / "box.json")
    n_own = sum(s.startswith("own") for s in status.values())
    devs = [float(np.linalg.norm(q - med, axis=1).max()) for q in good.values()]
    print(f"{len(vids)} clips: own detection kept for {n_own}, median fallback for {len(vids) - n_own}")
    print(f"detection spread vs median: median {np.median(devs):.1f}px, 90th {np.percentile(devs, 90):.1f}px, max {max(devs):.1f}px")
    print("session median corners:", med.round(1).tolist())
    print(f"sheet: {out / 'per_clip_sheet.jpg'}")


if __name__ == "__main__":
    main()
