"""Box calibration: four inner-floor corners in pixels -> homography to box metres.

The phone never moves between clips, so one calibration serves a whole session.
Corner order: c0 is the box origin, c1 is along the short side (size_x),
c2 is the opposite corner, c3 closes the loop along the long side (size_y).
Handedness does not matter physically (pushing is mirror-symmetric) but must
stay fixed across clips, which a single calibration guarantees.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BoxCalibration:
    corners_px: np.ndarray      # (4, 2)
    size_x: float
    size_y: float
    image_wh: tuple[int, int]
    H: np.ndarray = None        # pixel -> metres

    def __post_init__(self):
        self.corners_px = np.asarray(self.corners_px, dtype=np.float64).reshape(4, 2)
        dst = np.array([[0, 0], [self.size_x, 0], [self.size_x, self.size_y], [0, self.size_y]], dtype=np.float64)
        self.H, _ = cv2.findHomography(self.corners_px, dst)
        self.H_inv = np.linalg.inv(self.H)

    def to_box(self, px) -> np.ndarray:
        """Pixel coords (..., 2) -> box metres (..., 2)."""
        p = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(np.shape(px))

    def to_px(self, xy) -> np.ndarray:
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H_inv).reshape(np.shape(xy))

    def px_per_m(self) -> float:
        d = [np.linalg.norm(self.corners_px[(i + 1) % 4] - self.corners_px[i]) for i in range(4)]
        return float((d[0] + d[2]) / 2 / self.size_x + (d[1] + d[3]) / 2 / self.size_y) / 2

    def save(self, path):
        Path(path).write_text(json.dumps(dict(corners_px=self.corners_px.tolist(), size_x=self.size_x,
                                              size_y=self.size_y, image_wh=list(self.image_wh)), indent=1))

    @staticmethod
    def load(path) -> "BoxCalibration":
        d = json.loads(Path(path).read_text())
        return BoxCalibration(np.array(d["corners_px"]), d["size_x"], d["size_y"], tuple(d["image_wh"]))


def order_corners(quad: np.ndarray, size_x: float, size_y: float) -> np.ndarray:
    """Sort 4 points clockwise from the top-left, then rotate so edge c0->c1 is the short side."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    q = np.roll(q, -int(np.argmin(q.sum(1))), axis=0)
    e = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    first_short = (e[0] + e[2]) < (e[1] + e[3])
    if first_short != (size_x < size_y):
        q = np.roll(q, -1, axis=0)
    return q


def median_background(video_path, n_frames=25) -> np.ndarray:
    """Per-pixel median over frames spread across the clip: removes the moving hand."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(n - 1, 0), min(n_frames, max(n, 1))).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def fit_rim_lines(img, v_dark=100, min_votes=200, group_px=70, edge="outer"):
    """Primary detector: the four rim edges are the longest straight dark lines
    in the median background. Hough-fit them and intersect.

    ``edge="outer"`` picks, per side, the dark line farthest from the centre.
    The outer boundary of the rim is dark-against-white on all four sides,
    whereas the inner boundary is ambiguous where a wall's inner face is in
    shadow (it merges with the rim into one wide dark band). The rectangle is
    therefore the OUTER rim, i.e. inner floor size + 2 x wall thickness.
    Returns a (4, 2) quad or None."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dark = (hsv[..., 2] < v_dark).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(dark, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, min_votes)
    if lines is None:
        return None
    lines = lines[:, 0, :]  # (rho, theta), strongest first
    cx, cy = W / 2, H / 2
    groups = {"left": [], "right": [], "top": [], "bottom": []}
    for rho, th in lines:
        if abs(np.sin(th)) < 0.35:        # roughly x = const
            x_c = (rho - cy * np.sin(th)) / np.cos(th)
            groups["left" if x_c < cx else "right"].append((rho, th, x_c))
        elif abs(np.cos(th)) < 0.35:      # roughly y = const
            y_c = (rho - cx * np.cos(th)) / np.sin(th)
            groups["top" if y_c < cy else "bottom"].append((rho, th, y_c))
    sel = {}
    for k, g in groups.items():
        if not g:
            return None
        g = np.array(g)
        near = g[np.abs(g[:, 2] - g[0, 2]) < group_px]      # lines of the same rim edge
        want_min = (k in ("left", "top")) == (edge == "outer")
        pick = near[np.argmin(near[:, 2])] if want_min else near[np.argmax(near[:, 2])]
        sel[k] = pick[:2]

    def inter(l1, l2):
        A = np.array([[np.cos(l1[1]), np.sin(l1[1])], [np.cos(l2[1]), np.sin(l2[1])]])
        return np.linalg.solve(A, np.array([l1[0], l2[0]]))

    return np.array([inter(sel["top"], sel["left"]), inter(sel["top"], sel["right"]),
                     inter(sel["bottom"], sel["right"]), inter(sel["bottom"], sel["left"])])


def detect_box(img):
    """Hough rim fit; returns (quad or None, method). No silent fallback: a wrong
    rectangle is worse than none, so failure means clicking the corners once."""
    q = fit_rim_lines(img)
    if q is not None and np.all(q > -5) and np.all(q[:, 0] < img.shape[1] + 5) and np.all(q[:, 1] < img.shape[0] + 5):
        return q, "rim-lines"
    return None, "none"


def draw_calibration(img, calib: BoxCalibration, grid=0.05):
    vis = img.copy()
    pts = calib.corners_px.astype(int)
    cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    for k, p in enumerate(pts):
        cv2.circle(vis, tuple(p), 12, (0, 255, 255), -1)
        cv2.putText(vis, f"c{k}", tuple(p + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    for x in np.arange(0, calib.size_x + 1e-6, grid):
        a, b = calib.to_px([x, 0]), calib.to_px([x, calib.size_y])
        cv2.line(vis, tuple(a.astype(int)), tuple(b.astype(int)), (0, 200, 0), 1)
    for y in np.arange(0, calib.size_y + 1e-6, grid):
        a, b = calib.to_px([0, y]), calib.to_px([calib.size_x, y])
        cv2.line(vis, tuple(a.astype(int)), tuple(b.astype(int)), (0, 200, 0), 1)
    return vis
