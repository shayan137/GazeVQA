#!/usr/bin/env python3
"""Dense Gaze-LLE predictions for every benchmark person-frame -> cache.
Runs in the G3VQA environment; used to check the cached replay against the
live estimator."""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from tqdm.auto import tqdm

from benchmarks.common import CacheWriter, environment_meta, head_xyxy, iter_frames, load_frame_list, sha256_file
from source.tracking.estimators import GazeLLEEstimator, PersonContext


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="fkryan/gazelle")
    ap.add_argument("--model", default="gazelle_dinov2_vitl14_inout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frame-list rows")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", os.path.join("model_cache", "torch_hub"))
    frames = load_frame_list(args.frame_list)
    est = GazeLLEEstimator(args.repo, args.model, device=args.device)
    meta = {
        "model": "gazelle", "variant": args.model, "checkpoint": f"torch.hub {args.repo} {args.model}",
        "training_data": "GazeFollow + VideoAttentionTarget",
        "preprocessing": {"box": "ground-truth head box normalised to [0,1] (clipped)", "image": "448x448 squash, ImageNet normalisation"},
        "has_inout": True, "end_to_end": False,
        "frame_list": os.path.abspath(args.frame_list), "frame_list_sha256": sha256_file(args.frame_list),
        **environment_meta(),
    }
    writer = CacheWriter(args.out, frames, meta)
    if args.resume:
        print(f"[gazelle] resuming: {writer.resume()} rows already done")

    t0, n = time.time(), 0
    total = frames.iloc[: args.limit]["image_path"].nunique() if args.limit else frames["image_path"].nunique()
    for image, rows in tqdm(iter_frames(frames, limit=args.limit), total=total, desc="frames"):
        for _, row in rows.iterrows():
            i = int(row["row"])
            if writer.done[i]:
                continue
            box = head_xyxy(row)
            pred = est.predict(image, PersonContext(head_bbox_px=box))
            writer.add(i, pred.heatmap, pred.inout_score, model_box=box)
            n += 1
    writer.save()
    print(f"[gazelle] {n} predictions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
