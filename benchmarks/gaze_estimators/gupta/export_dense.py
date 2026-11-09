#!/usr/bin/env python3
"""Dense predictions of Gupta et al. (CVPRW 2022, RGB + depth + pose attention
model) for every benchmark person-frame -> cache. Input per call: 224x224
frame, 224x224 16-bit MiDaS DPT_Large depth / 65535
(benchmarks/aux/precompute_depth_midas.py), 224x224 skeleton image rendered
from torchvision KeypointRCNN keypoints
(benchmarks/aux/precompute_pose_keypointrcnn.py), face crop of the
ground-truth box expanded by 10 %, gaze field from the eye point and the head
mask. Output: 64x64 heatmap and the in/out head's sigmoid output (variant
"vat"); the "gazefollow" variant has no in/out head (inout = NaN)."""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm.auto import tqdm

from benchmarks.common import (CacheWriter, environment_meta, head_xyxy, iter_frames, load_frame_list,
                               overlay_debug_png, sha256_file)
from model_api import BOX_EXPAND_K, aux_paths, build, load_depth, load_pose, predict

CKPT_DIR = os.path.join(os.environ.get("GAZE_CKPTS", "gaze_ckpts"), "gupta")
CKPTS = {"vat": os.path.join(CKPT_DIR, "attention-videoatttarget.pt"),
         "gazefollow": os.path.join(CKPT_DIR, "attention-gazefollow.pt")}
HUMAN_CENTRIC = os.path.join(CKPT_DIR, "human-centric.pt")
TRAINING = {"vat": "GazeFollow -> VideoAttentionTarget (attention model, pred_inout)", "gazefollow": "GazeFollow (attention model, no in/out head)"}
DEVIATIONS = [
    "pose skeletons rendered from torchvision keypointrcnn_resnet50_fpn (COCO-17) instead of the repo's Faster R-CNN + HRNet-w48 (HRFormer/mmpose); same drawing code, person score >= 0.3, keypoint logit threshold instead of the HRNet 0.3 heatmap score",
    "depth from MiDaS DPT_Large via torch.hub (the repo uses MiDaS 'large' pt_large-midas-2f21e586.pt); 16-bit min-max PNG as in the repo's extraction",
    "cone_mode and pred_inout are inferred from the checkpoint weights (the released files lack those keys)",
    "16-bit depth PNG converted to PIL mode 'I' before ToTensor (the authors' Pillow 8.4 behaviour; current Pillow would wrap it to int16)",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="vat", choices=list(CKPTS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--aux-dir", default=os.path.join(ROOT, "outputs", "gaze_benchmark", "aux"))
    ap.add_argument("--device", default="cuda")  # the repo's forward hard-codes .cuda() for its dummy outputs
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()
    ckpt = args.checkpoint or CKPTS[args.variant]

    frames = load_frame_list(args.frame_list)
    model = build(ckpt, args.device)
    meta = {
        "model": "gupta", "variant": args.variant, "checkpoint": ckpt, "checkpoint_sha256": sha256_file(ckpt),
        "human_centric_checkpoint": HUMAN_CENTRIC, "human_centric_sha256": sha256_file(HUMAN_CENTRIC),
        "training_data": TRAINING[args.variant],
        "model_args": {"cone_mode": model.cone_mode_used, "pred_inout": model.pred_inout_used},
        "preprocessing": {"box": f"ground-truth head box expanded by k={BOX_EXPAND_K} (sequential, unclipped as in the VAT test loader), integer crop; eye point = (box centre x, 0.65*y1 + 0.35*y2) of the raw box",
                          "image": "224x224 squash + ImageNet normalisation; head mask 224; gaze field 2x224x224 unit vectors from the eye point",
                          "depth": "MiDaS DPT_Large 16-bit PNG (mode I) -> resize 224 -> /65535, 1 channel",
                          "pose": "KeypointRCNN skeleton JPEG (repo draw_pose colours) -> resize 224 -> ToTensor, 3 channels",
                          "inout": "in_vs_out head output (already sigmoid; used as-is by eval_on_videoatttarget.py)" if model.pred_inout_used else "none"},
        "aux": ["midas_dpt_large", "pose_keypointrcnn"], "deviations": DEVIATIONS,
        "has_inout": bool(model.pred_inout_used), "end_to_end": False,
        "note": "the in/out head ends with a Sigmoid and was trained with BCEWithLogitsLoss on that output (double sigmoid); ranking is unaffected",
        "frame_list": os.path.abspath(args.frame_list), "frame_list_sha256": sha256_file(args.frame_list),
        **environment_meta(),
    }
    writer = CacheWriter(args.out, frames, meta)
    if args.resume:
        print(f"[gupta] resuming: {writer.resume()} rows already done")

    t0, n, debug_left = time.time(), 0, 4 if args.debug_dir else 0
    total = frames.iloc[: args.limit]["image_path"].nunique() if args.limit else frames["image_path"].nunique()
    for image, rows in tqdm(iter_frames(frames, limit=args.limit), total=total, desc=f"gupta-{args.variant}"):
        depth_i = pose_rgb = None
        for _, row in rows.iterrows():
            i = int(row["row"])
            if writer.done[i]:
                continue
            if depth_i is None:
                dp, pp = aux_paths(args.aux_dir, row["image_path"])
                depth_i, pose_rgb = load_depth(dp), load_pose(pp)
            heatmap, inout, pbox = predict(model, image, depth_i, pose_rgb, head_xyxy(row), args.device)
            writer.add(i, heatmap, inout, model_box=pbox)
            n += 1
            if debug_left > 0:
                overlay_debug_png(image, pbox, heatmap, writer.point_norm[i], (row["gaze_x"], row["gaze_y"]),
                                  os.path.join(args.debug_dir, f"gupta_{args.variant}_{i:05d}.png"),
                                  title=f"{row['clip']} p{row['person_id']} f{row['frame']} inout={inout:.2f} gt={row['gaze_class']}")
                debug_left -= 1
    writer.save()
    print(f"[gupta] {n} predictions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
