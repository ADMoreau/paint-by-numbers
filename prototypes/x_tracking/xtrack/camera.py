"""Pinhole phone camera looking at the canvas, and camera poses.

Canvas coordinates are millimetres on the canvas plane z = 0: x to the right,
y down, origin at the top-left corner. z points into the canvas, so a camera
in front of it sits at negative z and looks along +z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int = 1280
    height: int = 960
    hfov_deg: float = 66.0  # typical main phone camera

    @property
    def f(self) -> float:
        return (self.width / 2) / np.tan(np.radians(self.hfov_deg) / 2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0.0, (self.width - 1) / 2], [0.0, self.f, (self.height - 1) / 2], [0.0, 0.0, 1.0]])

    def view_mm(self, distance: float) -> tuple[float, float]:
        """Width and height of canvas seen head-on from ``distance`` mm."""
        return distance * self.width / self.f, distance * self.height / self.f


MAIN = Camera()
ULTRAWIDE = Camera(hfov_deg=105.0)


def rotation(axis, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` radians about ``axis``."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


@dataclass(frozen=True)
class Pose:
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation, mm

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


def look_at(target_xy, distance: float, tilt_deg: float = 0.0, tilt_dir_deg: float = 0.0, roll_deg: float = 0.0) -> Pose:
    """A camera whose optical axis hits ``target_xy`` from ``distance`` mm,
    tilted ``tilt_deg`` away from head-on and rolled ``roll_deg`` about its axis."""
    a = np.radians(tilt_dir_deg)
    R_wc = rotation([np.cos(a), np.sin(a), 0.0], np.radians(tilt_deg)) @ rotation([0, 0, 1], np.radians(roll_deg))
    C = np.array([target_xy[0], target_xy[1], 0.0]) - distance * R_wc[:, 2]
    R = R_wc.T
    return Pose(R, -R @ C)


def homography(cam: Camera, pose: Pose) -> np.ndarray:
    """Canvas mm (on z = 0) -> image pixels."""
    H = cam.K @ np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.t])
    return H / np.linalg.norm(H)  # scale only: keeps w > 0 for points in front of the camera


def in_front(H: np.ndarray, pts) -> np.ndarray:
    """Which canvas points are in front of the camera (for H from ``homography``)."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    return pts @ H[2, :2] + H[2, 2] > 1e-9


def project(H: np.ndarray, pts) -> np.ndarray:
    """Apply a homography to (n, 2) points."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = pts @ H[:, :2].T + H[:, 2]
    return p[:, :2] / p[:, 2:3]


def local_scale(H: np.ndarray, xy) -> float:
    """Pixels per mm near canvas point ``xy``."""
    x, y = xy
    p = project(H, [(x, y), (x + 1, y), (x, y + 1)])
    return float(np.sqrt(abs(np.linalg.det(np.column_stack([p[1] - p[0], p[2] - p[0]])))))


def random_pose(rng, canvas_size, distance: float, max_tilt_deg: float = 25.0) -> Pose:
    """Aimed anywhere on the canvas, any rotation, tilted up to ``max_tilt_deg``."""
    target = rng.uniform((0, 0), canvas_size)
    tilt = max_tilt_deg * np.sqrt(rng.uniform())
    return look_at(target, distance, tilt, rng.uniform(0, 360), rng.uniform(0, 360))
