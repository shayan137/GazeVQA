"""Lookup of predicted gaze points for a question.

Person-specific questions get one arrow for that person, scene-level
questions ("all") get one arrow per visible person, and ambiguous-description
questions get the plain frame, since which person is meant is what the
question tests.
"""

from pathlib import Path

import pandas as pd

from source.data.vqa import AMBIGUOUS_IDS, SCENE_IDS
from source.vqa.visual_prompt import head_center

REQUIRED_COLUMNS = {
    "clip", "person_id", "frame", "image_path",
    "head_x1", "head_y1", "head_x2", "head_y2", "point_x", "point_y",
}


class GazePredictions:
    def __init__(self, predictions_csv, frames_dir=None, clips: list[str] | None = None):
        df = pd.read_csv(predictions_csv, dtype={"person_id": str})
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{predictions_csv} is missing columns {sorted(missing)}")
        if clips is not None:
            df = df[df["clip"].isin(clips)]
        if frames_dir is not None:
            # image paths in the CSV are absolute paths from the machine that ran tracking
            df["image_path"] = [str(Path(frames_dir) / c / Path(p).name) for c, p in zip(df["clip"], df["image_path"])]

        self.by_person = {}
        self.by_frame = {}
        for row in df.itertuples():
            self.by_person[(row.clip, row.person_id, int(row.frame))] = row
            self.by_frame.setdefault((row.clip, int(row.frame)), []).append(row)

    @staticmethod
    def _arrow(p):
        return head_center(p.head_x1, p.head_y1, p.head_x2, p.head_y2), (p.point_x, p.point_y)

    def lookup(self, clip: str, person_id: str, frame: int):
        """Returns (image_path, arrows, has_overlay, is_scene). image_path is
        None when there is no usable gaze point for this question."""
        frame = int(frame)
        if person_id in AMBIGUOUS_IDS:
            rows = self.by_frame.get((clip, frame))
            return (rows[0].image_path if rows else None), [], False, False

        if person_id in SCENE_IDS:
            people = [p for p in self.by_frame.get((clip, frame), []) if not pd.isna(p.point_x)]
            image_path = people[0].image_path if people else None
            return image_path, [self._arrow(p) for p in people], True, True

        p = self.by_person.get((clip, person_id, frame))
        if p is None or pd.isna(p.point_x):
            return None, [], True, False
        return p.image_path, [self._arrow(p)], True, False
