#!/usr/bin/env python3
"""Dense PaGE predictions (released HF checkpoint, trust_remote_code) for every
benchmark person-frame -> cache. Input per call: the frame plus the
ground-truth head box, exactly as scripts/inference.py in the PaGE repo
(integer pixel box, head crop = scene.crop(box), normalised box for the head
prompt). Output: 64x64 sigmoid heatmap and P(in frame)."""
import argparse
import contextlib
import io
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from tqdm.auto import tqdm

from benchmarks.common import (CacheWriter, environment_meta, head_xyxy, iter_frames, load_frame_list,
                               overlay_debug_png, sha256_file)


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def load_model(model_path: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    with contextlib.redirect_stdout(io.StringIO()):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(device)
        processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def pixel_box(box, W: int, H: int):
    """Same rounding/clipping as PaGE's scripts/inference.py parse_bbox."""
    x1 = max(0, min(W - 1, int(round(box[0]))))
    y1 = max(0, min(H - 1, int(round(box[1]))))
    x2 = max(0, min(W, int(round(box[2]))))
    y2 = max(0, min(H, int(round(box[3]))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty head box {box}")
    return (x1, y1, x2, y2), (x1 / W, y1 / H, x2 / W, y2 / H)


def build_inputs(processor, image, box, device):
    pbox, nbox = pixel_box(box, *image.size)
    inputs = processor(image, head_crops=[image.crop(pbox)], bboxes=[[nbox]])
    return move_to_device(inputs, device), pbox


def predict(model, processor, image, box, device):
    inputs, pbox = build_inputs(processor, image, box, device)
    with torch.inference_mode():
        out = model(inputs)
    heatmap = out["heatmap"][0][0].detach().float().cpu().numpy()  # (64, 64) sigmoid probabilities
    inout = float("nan")
    if out.get("inout") is not None:
        inout = float(out["inout"][0].detach().float().cpu().reshape(-1)[0])
    return heatmap, inout, pbox


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="Octopus1/page-vitb", help="HF repo id or local checkpoint dir")
    ap.add_argument("--variant", default="vitb")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--debug-dir", default=None, help="write overlays of the first rows here")
    args = ap.parse_args()

    frames = load_frame_list(args.frame_list)
    model, processor = load_model(args.model_path, args.device)
    meta = {
        "model": "page", "variant": args.variant, "checkpoint": args.model_path,
        "training_data": "GazeFollow + VideoAttentionTarget + ChildPlay (joint)",
        "preprocessing": {"box": "ground-truth head box rounded to integer pixels and clipped; head crop = scene.crop(box); normalised box as head prompt",
                          "image": "PaGE HF processor (scene 512, head 256)"},
        "has_inout": True, "end_to_end": False,
        "frame_list": os.path.abspath(args.frame_list), "frame_list_sha256": sha256_file(args.frame_list),
        **environment_meta(),
    }
    writer = CacheWriter(args.out, frames, meta)
    if args.resume:
        print(f"[page] resuming: {writer.resume()} rows already done")

    t0, n, debug_left = time.time(), 0, 4 if args.debug_dir else 0
    total = frames.iloc[: args.limit]["image_path"].nunique() if args.limit else frames["image_path"].nunique()
    for image, rows in tqdm(iter_frames(frames, limit=args.limit), total=total, desc=f"page-{args.variant}"):
        for _, row in rows.iterrows():
            i = int(row["row"])
            if writer.done[i]:
                continue
            box = head_xyxy(row)
            heatmap, inout, pbox = predict(model, processor, image, box, args.device)
            writer.add(i, heatmap, inout, model_box=pbox)
            n += 1
            if debug_left > 0:
                overlay_debug_png(image, pbox, heatmap, writer.point_norm[i], (row["gaze_x"], row["gaze_y"]),
                                  os.path.join(args.debug_dir, f"page_{args.variant}_{i:05d}.png"),
                                  title=f"{row['clip']} p{row['person_id']} f{row['frame']} inout={inout:.2f} gt={row['gaze_class']}")
                debug_left -= 1
    writer.save()
    print(f"[page] {n} predictions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
