#!/usr/bin/env python3
"""Dense GazeAnywhere predictions for every person-frame -> cache.
The text prompt is the person's appearance description (propagated over the track in the
frame list); person-frames without any description use "the person". Runs in the G3VQA
(childplay) environment."""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from tqdm.auto import tqdm

from benchmarks.common import CacheWriter, environment_meta, head_xyxy, iter_frames, load_frame_list, sha256_file
from source.tracking.estimators import GazeAnywhereEstimator, PersonContext


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="IrohXu/GazeAnywhere")
    ap.add_argument("--repo-dir", default=os.environ.get("GAZE_ANYWHERE_REPO", ""))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    frames = load_frame_list(args.frame_list)
    est = GazeAnywhereEstimator(args.model, args.repo_dir, device=args.device)
    meta = {
        "model": "gaze_anywhere", "variant": args.model, "checkpoint": f"HF {args.model}",
        "training_data": "GazeAnywhere release",
        "preprocessing": {"prompt": "per-track appearance description (look_like), 'the person' if missing"},
        "has_inout": True, "end_to_end": False,
        "frame_list": os.path.abspath(args.frame_list), "frame_list_sha256": sha256_file(args.frame_list),
        **environment_meta(),
    }
    writer = CacheWriter(args.out, frames, meta, flush_every=300)
    if args.resume:
        print(f"[gaze_anywhere] resuming: {writer.resume()} rows already done")

    t0, n = time.time(), 0
    total = frames.iloc[: args.limit]["image_path"].nunique() if args.limit else frames["image_path"].nunique()
    for image, rows in tqdm(iter_frames(frames, limit=args.limit), total=total, desc="frames"):
        for _, row in rows.iterrows():
            i = int(row["row"])
            if writer.done[i]:
                continue
            pred = est.predict(image, PersonContext(head_bbox_px=head_xyxy(row), description=row.get("look_like")))
            writer.add(i, pred.heatmap, pred.inout_score, model_box=head_xyxy(row))
            n += 1
    writer.save()
    print(f"[gaze_anywhere] {n} predictions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
