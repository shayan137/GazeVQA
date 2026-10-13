"""Lucas-Kanade tracking of a single gaze point between trigger frames.

The gaze point itself often lies on a low-texture region, so we track a few
corner features around it and move the point by their median displacement.
A forward-backward check drops features that flow onto the wrong patch.
"""

import cv2
import numpy as np

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FB_ERROR_THRESHOLD_PX = 8.0
SEARCH_RADIUS_PX = 30
MAX_CORNERS = 8


class PointTracker:
    def __init__(self):
        self._prev_gray = None
        self._point = None

    def init(self, frame_bgr: np.ndarray, point_xy) -> tuple[float, float]:
        self._prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._point = (float(point_xy[0]), float(point_xy[1]))
        return self._point

    def _local_features(self, gray: np.ndarray, point) -> np.ndarray | None:
        h, w = gray.shape[:2]
        r = SEARCH_RADIUS_PX
        x0, y0 = int(max(0, point[0] - r)), int(max(0, point[1] - r))
        x1, y1 = int(min(w, point[0] + r)), int(min(h, point[1] + r))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        corners = cv2.goodFeaturesToTrack(gray[y0:y1, x0:x1], maxCorners=MAX_CORNERS, qualityLevel=0.01, minDistance=4)
        if corners is None:
            return None
        return corners.reshape(-1, 2).astype(np.float32) + np.array([x0, y0], dtype=np.float32)

    def update(self, frame_bgr: np.ndarray) -> tuple[tuple[float, float], bool]:
        """Returns the new point and whether tracking succeeded. On failure the
        previous point is kept."""
        if self._prev_gray is None or self._point is None:
            return self._point, False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        features = self._local_features(self._prev_gray, self._point)
        if features is None or len(features) == 0:
            self._prev_gray = gray
            return self._point, False

        pts = features.reshape(-1, 1, 2)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, pts, None, **LK_PARAMS)
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, nxt, None, **LK_PARAMS)
        self._prev_gray = gray

        good = (status.reshape(-1) == 1) & (back_status.reshape(-1) == 1)
        if good.any():
            fb_error = np.linalg.norm(back.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
            good &= fb_error <= FB_ERROR_THRESHOLD_PX
        if not good.any():
            return self._point, False

        dx, dy = np.median((nxt.reshape(-1, 2) - pts.reshape(-1, 2))[good], axis=0)
        h, w = gray.shape[:2]
        x = min(max(self._point[0] + float(dx), 0.0), w - 1.0)
        y = min(max(self._point[1] + float(dy), 0.0), h - 1.0)
        self._point = (x, y)
        return self._point, True
