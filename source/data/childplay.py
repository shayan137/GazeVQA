"""Per-clip label files in the ChildPlay format.

One row per (frame, person): head box, gaze class, gaze point and the
person's appearance description ("look like"). Frames are clip-relative
and 1-indexed, stored as <frames_dir>/<clip>/<clip>_<frame:06d>.jpg.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BBOX_COLUMNS = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]


def frame_path(frames_dir, clip: str, frame: int) -> Path:
    return Path(frames_dir) / clip / f"{clip}_{int(frame):06d}.jpg"


def _read_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [" ".join(c.strip().lower().split()) for c in df.columns]
    return df.apply(lambda col: col.str.strip())


def load_labels(labels_dir, frames_dir) -> pd.DataFrame:
    paths = sorted(p for p in Path(labels_dir).glob("*.csv") if not p.stem.lower().endswith("_vqa"))
    if not paths:
        raise FileNotFoundError(f"no label CSVs found in {labels_dir}")
    rows = pd.concat([_read_label_csv(p) for p in paths], ignore_index=True)

    for col in ("gaze_class", "look like"):
        if col not in rows:
            rows[col] = ""
    rows = rows.rename(columns={"look like": "look_like"})
    rows["gaze_class"] = rows["gaze_class"].replace("", pd.NA)
    rows["look_like"] = rows["look_like"].replace("", pd.NA)

    rows["frame"] = pd.to_numeric(rows["frame"]).astype(int)
    for col in BBOX_COLUMNS + ["gaze_x", "gaze_y"]:
        rows[col] = pd.to_numeric(rows.get(col, ""), errors="coerce")

    bbox = rows[BBOX_COLUMNS].to_numpy(dtype=float)
    rows["bbox_valid"] = np.isfinite(bbox).all(axis=1) & (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

    rows["image_path"] = [str(frame_path(frames_dir, c, f)) for c, f in zip(rows["clip"], rows["frame"])]
    rows["image_exists"] = [Path(p).is_file() for p in rows["image_path"]]
    return rows
