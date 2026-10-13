#!/usr/bin/env python3
"""
run the gaze tracking agent on every person track: OpenFace finds the
trigger frames, the gaze estimator runs on those, optical flow follows the
gaze point in between. With --dense the estimator runs on every frame instead.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from tqdm.auto import tqdm

from source.config import TrackingConfig
from source.data.childplay import load_labels
from source.tracking.agent import OUTPUT_COLUMNS, GazeTrackingAgent
from source.tracking.estimators import build_estimator
from source.tracking.triggers import find_triggers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to configs/tracking.yaml")
    ap.add_argument("--labels-dir", default=None, help="overrides cfg.labels_dir")
    ap.add_argument("--frames-dir", default=None, help="overrides cfg.frames_dir")
    ap.add_argument("--output-csv", default=None, help="overrides cfg.output_csv")
    ap.add_argument("--estimator", default=None, choices=["gazelle", "gaze_anywhere"], help="overrides cfg.estimator")
    ap.add_argument("--dense", action="store_true", help="run the estimator on every frame (baseline)")
    ap.add_argument("--clips", default=None, help="comma-separated clip names, default all")
    ap.add_argument("--resume", action="store_true", help="skip tracks already in the output CSV")
    args = ap.parse_args()

    cfg = TrackingConfig.from_yaml(args.config)
    for key in ("labels_dir", "frames_dir", "output_csv", "estimator"):
        if getattr(args, key):
            setattr(cfg, key, getattr(args, key))
    cfg.dense = cfg.dense or args.dense

    rows = load_labels(cfg.labels_dir, cfg.frames_dir)
    if args.clips:
        rows = rows[rows["clip"].isin(args.clips.split(","))]
    print(f"[tracking] {len(rows)} person-frames, {rows.groupby(['clip', 'person_id']).ngroups} tracks")

    triggers = None if cfg.dense else find_triggers(rows, cfg)
    rows = rows[rows["bbox_valid"] & rows["image_exists"]]

    agent = GazeTrackingAgent(build_estimator(cfg), cfg.inout_threshold, cfg.drift_head_diagonals)
    print(f"[tracking] estimator: {cfg.estimator}, mode: {'dense' if cfg.dense else 'sparse'}")

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_csv)), exist_ok=True)
    done = set()
    if args.resume and os.path.isfile(cfg.output_csv):
        prev = pd.read_csv(cfg.output_csv, usecols=["clip", "person_id"], dtype={"person_id": str})
        done = set(prev.drop_duplicates().itertuples(index=False, name=None))
    header = not done

    for (clip, pid), track in tqdm(list(rows.groupby(["clip", "person_id"], sort=False)), desc="tracks"):
        if (clip, pid) in done:
            continue
        if cfg.dense:
            out = agent.run_track_dense(track)
        else:
            frames = triggers.loc[(triggers["clip"] == clip) & (triggers["person_id"] == pid), "frame"]
            out = agent.run_track(track, frames.tolist())
        pd.DataFrame(out, columns=OUTPUT_COLUMNS).to_csv(
            cfg.output_csv, mode="a" if not header else "w", header=header, index=False
        )
        header = False

    print(f"[tracking] predictions written to {cfg.output_csv}")


if __name__ == "__main__":
    main()
