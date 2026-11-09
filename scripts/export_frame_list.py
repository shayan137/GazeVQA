#!/usr/bin/env python3
"""
write the person-frame list the gaze-estimator benchmark runs on: one row per
(clip, person_id, frame) with image path and size, head box and ground truth,
in a fixed order that every dense prediction cache follows
"""
import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from PIL import Image

from source.data.childplay import load_labels, propagate_look_like


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--clips", default="", help="comma-separated clip names, default all clips in --labels-dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_labels(args.labels_dir, args.frames_dir)
    if args.clips:
        rows = rows[rows["clip"].isin(args.clips.split(","))]
    rows = propagate_look_like(rows)
    rows = rows[rows["bbox_valid"] & rows["image_exists"]].copy()
    rows["person_order"] = pd.to_numeric(rows["person_id"], errors="coerce")
    rows = rows.sort_values(["clip", "person_order", "person_id", "frame"], kind="mergesort")

    sizes = {}
    for path in rows["image_path"].unique():
        with Image.open(path) as im:
            sizes[path] = im.size
    out = pd.DataFrame({
        "clip": rows["clip"].to_numpy(), "person_id": rows["person_id"].to_numpy(), "frame": rows["frame"].to_numpy(),
        "image_path": rows["image_path"].to_numpy(),
        "W": [sizes[p][0] for p in rows["image_path"]], "H": [sizes[p][1] for p in rows["image_path"]],
        "head_x1": rows["bbox_x"].to_numpy(), "head_y1": rows["bbox_y"].to_numpy(),
        "head_x2": (rows["bbox_x"] + rows["bbox_width"]).to_numpy(), "head_y2": (rows["bbox_y"] + rows["bbox_height"]).to_numpy(),
        "gaze_class": rows["gaze_class"].to_numpy(), "gaze_x": rows["gaze_x"].to_numpy(), "gaze_y": rows["gaze_y"].to_numpy(),
        "look_like": rows["look_like"].to_numpy(),
    })
    if out.duplicated(["clip", "person_id", "frame"]).any():
        raise ValueError("duplicate (clip, person_id, frame) rows in the labels")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False)
    with open(args.out, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(args.out + ".sha256", "w") as f:
        f.write(sha + "\n")
    print(f"[frame-list] {len(out)} person-frames, {out.groupby(['clip', 'person_id']).ngroups} tracks, "
          f"{out['image_path'].nunique()} unique frames, {out['clip'].nunique()} clips -> {args.out} (sha256 {sha[:12]})")


if __name__ == "__main__":
    main()
