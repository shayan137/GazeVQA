"""Trigger frames: where the gaze estimator has to be (re-)run.

Each track is triggered on its first frame and on every frame where the
OpenFace head/eye direction changes by more than a threshold between
consecutive frames. Consecutive above-threshold frames form one bout, and
only its peak frame is kept.
"""

import numpy as np
import pandas as pd

from source.tracking.openface import head_pose, load_openface


def to_unit_vectors(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    cos_p = np.cos(pitch)
    v = np.stack([cos_p * np.sin(yaw), np.sin(pitch), cos_p * np.cos(yaw)], axis=-1)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.full_like(v, np.nan), where=np.isfinite(norm) & (norm > 0))


def angular_steps(poses: pd.DataFrame) -> pd.DataFrame:
    """Adds angular_step_deg: angle between the direction at frame t-1 and t,
    NaN at the start of every contiguous valid run."""
    poses = poses.sort_values(["clip", "person_id", "frame"], kind="mergesort").reset_index(drop=True)
    v = to_unit_vectors(poses["yaw"].to_numpy(float), poses["pitch"].to_numpy(float))
    frames = poses["frame"].to_numpy(float)
    valid = poses["valid"].to_numpy(bool) & np.isfinite(v).all(axis=1)
    same_track = (poses["clip"].eq(poses["clip"].shift()) & poses["person_id"].eq(poses["person_id"].shift())).to_numpy()

    steps = np.full(len(poses), np.nan)
    linked = np.zeros(len(poses), dtype=bool)
    linked[1:] = same_track[1:] & valid[1:] & valid[:-1] & (np.diff(frames) == 1)
    dots = np.clip(np.einsum("ij,ij->i", v[1:], v[:-1]), -1.0, 1.0)
    steps[1:][linked[1:]] = np.degrees(np.arccos(dots[linked[1:]]))
    poses["angular_step_deg"] = steps
    return poses


def _track_events(frames: np.ndarray, scores: np.ndarray, threshold: float, min_peak_distance: int) -> list:
    valid = np.isfinite(scores)
    events = []
    i, n = 0, len(frames)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        end = i + 1
        while end < n and valid[end] and frames[end] - frames[end - 1] == 1:
            end += 1

        # one candidate per bout of above-threshold steps, at its peak
        candidates = []
        j = i
        while j < end:
            if scores[j] < threshold:
                j += 1
                continue
            k = j + 1
            while k < end and scores[k] >= threshold:
                k += 1
            candidates.append(j + int(np.argmax(scores[j:k])))
            j = k

        # within a segment, keep stronger peaks at least min_peak_distance apart
        selected = []
        for c in sorted(candidates, key=lambda c: (-scores[c], frames[c], c)):
            if all(abs(frames[c] - frames[s]) >= min_peak_distance for s in selected):
                selected.append(c)
        events.extend(sorted(selected))
        i = end
    return [frames[c] for c in events]


def detect_events(steps: pd.DataFrame, threshold: float, min_peak_distance: int) -> pd.DataFrame:
    events = []
    for (clip, pid), track in steps.groupby(["clip", "person_id"], sort=False):
        frames = track["frame"].to_numpy(float)
        for f in _track_events(frames, track["angular_step_deg"].to_numpy(float), threshold, min_peak_distance):
            events.append({"clip": clip, "person_id": pid, "frame": int(f)})
    return pd.DataFrame(events, columns=["clip", "person_id", "frame"])


def find_triggers(rows: pd.DataFrame, cfg) -> pd.DataFrame:
    """One row per (clip, person_id, frame) where the estimator should run."""
    model = load_openface(cfg.openface_root, cfg.openface_checkpoint, device=cfg.device)
    poses = head_pose(rows, model, device=cfg.device)
    del model
    if cfg.device != "cpu":
        import torch
        torch.cuda.empty_cache()

    events = detect_events(angular_steps(poses), cfg.trigger_threshold_deg, cfg.min_peak_distance)
    first = rows.sort_values(["clip", "person_id", "frame"]).groupby(["clip", "person_id"], as_index=False).first()
    return (
        pd.concat([first[["clip", "person_id", "frame"]], events], ignore_index=True)
        .drop_duplicates()
        .sort_values(["clip", "person_id", "frame"])
        .reset_index(drop=True)
    )
