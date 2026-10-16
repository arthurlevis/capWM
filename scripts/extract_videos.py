"""Extract trajectories from every video in a folder.

    python scripts/extract_videos.py --videos data/raw --calib data/calib/box.json --out data/processed/video --overlay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pushwm.config import load_config
from pushwm.vision.calib import BoxCalibration
from pushwm.vision.extract import extract_trajectory

EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default="data/raw")
    p.add_argument("--calib", default="data/calib/box.json")
    p.add_argument("--out", default="data/processed/video")
    p.add_argument("--overlay", action="store_true", help="also write tracking overlay videos")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--cache", default="data/processed/tracks", help="per-frame pixel tracks, reused on recalibration")
    a = p.parse_args()
    cfg = load_config(a.config)
    calib = BoxCalibration.load(a.calib)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ov_dir = out / "overlays"; ov_dir.mkdir(exist_ok=True)
    cache_dir = Path(a.cache); cache_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted(p for p in Path(a.videos).iterdir() if p.suffix.lower() in EXT)
    report = {}
    for v in vids:
        traj, stats = extract_trajectory(v, calib, cfg.control.dt,
                                         overlay_path=(ov_dir / f"{v.stem}.mp4") if a.overlay else None,
                                         max_frames=a.max_frames, track_cache=cache_dir / f"{v.stem}.npz")
        traj.save(out / f"{v.stem}.npz")
        report[v.name] = stats
        print(f"{v.name}: frames={stats['frames']} fps={stats['fps']:.1f} tracked={stats['frame_valid']:.0%} "
              f"-> steps={stats['steps']} valid={stats['step_valid']:.0%}")
    (out / "extraction_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(vids)} videos -> {out}")


if __name__ == "__main__":
    main()
