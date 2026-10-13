"""Training-free gaze tracking agent for video.

For each person track, the image gaze estimator runs only on trigger frames
(first frame, OpenFace head/eye movement events). In between, the predicted
gaze point is followed with optical flow. The estimator is also re-run when
the tracked point drifts more than `drift_head_diagonals` head diagonals
from the last estimate, which catches gaze shifts with little head motion.
"""

import numpy as np
import pandas as pd
from PIL import Image

from source.tracking import metrics
from source.tracking.estimators import PersonContext
from source.tracking.point_tracker import PointTracker

OUTPUT_COLUMNS = [
    "clip", "person_id", "frame", "image_path",
    "head_x1", "head_y1", "head_x2", "head_y2",
    "point_x", "point_y", "inout_score", "in_frame",
    "is_trigger_frame", "track_lost",
    "gt_evaluable", "inout_gt", "auc", "l2",
]


def head_bbox(row) -> tuple[float, float, float, float]:
    x, y, w, h = row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"]
    return x, y, x + w, y + h


def to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image)[:, :, ::-1].copy()


class GazeTrackingAgent:
    def __init__(self, estimator, inout_threshold: float = 0.35, drift_head_diagonals: float = 1.0):
        self.estimator = estimator
        self.inout_threshold = inout_threshold
        self.drift_head_diagonals = drift_head_diagonals

    def _record(self, row, image, bbox, point, inout, in_frame, trigger, lost, heatmap) -> dict:
        return {
            "clip": row["clip"], "person_id": row["person_id"], "frame": int(row["frame"]),
            "image_path": row["image_path"],
            "head_x1": bbox[0], "head_y1": bbox[1], "head_x2": bbox[2], "head_y2": bbox[3],
            "point_x": point[0], "point_y": point[1], "inout_score": inout, "in_frame": in_frame,
            "is_trigger_frame": trigger, "track_lost": lost,
            **metrics.score_frame(row, image.size, point, heatmap),
        }

    def run_track(self, track_rows: pd.DataFrame, trigger_frames) -> list[dict]:
        triggers = set(int(f) for f in trigger_frames)
        tracker = PointTracker()
        have_point = False
        anchor = None  # last point confirmed by the estimator
        out = []

        for _, row in track_rows.sort_values("frame").iterrows():
            image = Image.open(row["image_path"]).convert("RGB")
            bbox = head_bbox(row)
            head_diag = float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            trigger = int(row["frame"]) in triggers

            point = (float("nan"), float("nan"))
            inout = float("nan")
            in_frame = lost = False
            heatmap = None

            if not trigger and have_point:
                point, ok = tracker.update(to_bgr(image))
                lost = not ok
                if anchor is not None and head_diag > 0:
                    drift = float(np.hypot(point[0] - anchor[0], point[1] - anchor[1]))
                    trigger = drift > self.drift_head_diagonals * head_diag

            if trigger:
                person = PersonContext(head_bbox_px=bbox, description=row.get("look_like"))
                pred = self.estimator.predict(image, person)
                point, inout, heatmap = (pred.x_px, pred.y_px), pred.inout_score, pred.heatmap
                in_frame = inout >= self.inout_threshold
                if in_frame:
                    point = tracker.init(to_bgr(image), point)
                    have_point, anchor = True, point
                else:
                    # looking outside the frame: nothing to track or score AUC on
                    have_point, anchor, heatmap = False, None, None

            out.append(self._record(row, image, bbox, point, inout, in_frame, trigger, lost, heatmap))
        return out

    def run_track_dense(self, track_rows: pd.DataFrame) -> list[dict]:
        """Baseline: run the estimator on every frame."""
        out = []
        for _, row in track_rows.sort_values("frame").iterrows():
            image = Image.open(row["image_path"]).convert("RGB")
            bbox = head_bbox(row)
            pred = self.estimator.predict(image, PersonContext(head_bbox_px=bbox, description=row.get("look_like")))
            in_frame = pred.inout_score >= self.inout_threshold
            out.append(self._record(
                row, image, bbox, (pred.x_px, pred.y_px), pred.inout_score, in_frame, True, False, pred.heatmap
            ))
        return out
