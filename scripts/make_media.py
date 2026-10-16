"""Build the tracking GIFs for the README from the overlay videos.

    python scripts/make_media.py
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def gif(src, dst, fps=10, width=640, start=None, dur=None):
    pre = ([] if start is None else ["-ss", str(start)]) + ([] if dur is None else ["-t", str(dur)])
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *pre, "-i", str(src), "-vf", vf, str(dst)], check=True)
    print(dst, f"{Path(dst).stat().st_size/1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/media")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [("data/processed/video/overlays/IMG_0510.mp4", "tracking_push_clip.gif", 360, 2.5, 5),
            ("data/processed/video/overlays/IMG_0527.mp4", "tracking_push_clip_2.gif", 360, 3.0, 8),
            ("data/processed/video/overlays/IMG_0546.mp4", "tracking_nocontact_clip.gif", 360, 1.5, 3)]
    for src, dst, w, s, d in jobs:
        if Path(src).exists():
            gif(src, out / dst, width=w, start=s, dur=d)
        else:
            print("missing", src)


if __name__ == "__main__":
    main()
