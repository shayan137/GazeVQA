#!/usr/bin/env python3
"""Dense Sharingan predictions (CVPR 2024) for every benchmark person-frame ->
cache. Mirrors demo.py: load_sharingan_model (L111-138, constructor =
src/conf/config_cp.yaml / config_vat.yaml model.sharingan, "model." prefix
stripped, strict load) and predict_gaze steps 3-7 (L163-196) with the
ground-truth head box of the target person instead of detections: squared box
(src/utils/common.py square_bbox), 224x224 head crop and 224x224 squashed
scene with IMG_MEAN/IMG_STD, normalised box clamped to [0, 1] as the repo's
test loaders do (src/datasets/childplay.py L182-192). Only the target person
is given to the model, so it is the last (and only) token like in
SharinganModule.test_step (L368-371). Output: raw 64x64 heatmap and
P(in frame) = sigmoid(in/out logit) (the inout decoder has use_sigmoid=False)."""
import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

from benchmarks.common import (CacheWriter, environment_meta, expand_box, head_xyxy, iter_frames, load_frame_list,
                               overlay_debug_png, sha256_file, sigmoid)

REPO = os.environ.get("SHARINGAN_REPO", "sharingan-main")
CKPT_DIR = os.path.join(os.environ.get("GAZE_CKPTS", "gaze_ckpts"), "sharingan", "checkpoints")
VARIANTS = {  # variant -> (checkpoint, training data)
    "childplay": (f"{CKPT_DIR}/childplay.pt", "GazeFollow (pre-training) -> ChildPlay (fine-tuned)"),
    "vat": (f"{CKPT_DIR}/videoattentiontarget.pt", "GazeFollow (pre-training) -> VideoAttentionTarget (fine-tuned)"),
    "gazefollow": (f"{CKPT_DIR}/gazefollow.pt", "GazeFollow"),
}
IMG_MEAN = [0.44232, 0.40506, 0.36457]  # demo.py L64-65
IMG_STD = [0.28674, 0.27776, 0.27995]
IMAGE_SIZE, HEAD_SIZE = 224, 224  # config_cp/config_vat data.image_size; head crops 224 (demo L168, datasets)
MODEL_ARGS = dict(  # demo.py L113-130 == src/conf/config_cp.yaml and config_vat.yaml model.sharingan
    patch_size=16, token_dim=768, image_size=IMAGE_SIZE, gaze_feature_dim=512, encoder_depth=12, encoder_num_heads=12,
    encoder_num_global_tokens=0, encoder_mlp_ratio=4.0, encoder_use_qkv_bias=True, encoder_drop_rate=0.0,
    encoder_attn_drop_rate=0.0, encoder_drop_path_rate=0.0, decoder_feature_dim=128, decoder_hooks=[2, 5, 8, 11],
    decoder_hidden_dims=[48, 96, 192, 384], decoder_use_bn=True,
)


def square_bbox(box):
    """src/utils/common.py square_bbox (L109-148): square around the box centre, side = max(w, h), NOT clipped."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy, s = x1 + w / 2, y1 + h / 2, max(w, h)
    return (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2)


def load_model(checkpoint: str, device: str, repo: str = REPO):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.modeling.sharingan import Sharingan

    model = Sharingan(**MODEL_ARGS)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {name.replace("model.", ""): value for name, value in ckpt["state_dict"].items()}  # demo.py L133-134
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def build_inputs(image, box, device: str, expand: float = 0.0, clamp_box: bool = True):
    """demo.py predict_gaze steps 3-6 (L163-186) for the single target person."""
    W, H = image.size
    if expand > 0:
        box = expand_box(box, expand, W, H)
    sbox = square_bbox(box)
    head = TF.resize(TF.to_tensor(image.crop(sbox)), (HEAD_SIZE, HEAD_SIZE), antialias=True)  # PIL rounds, black padding
    heads = TF.normalize(head.unsqueeze(0), mean=IMG_MEAN, std=IMG_STD)
    img = TF.resize(TF.to_tensor(image), (IMAGE_SIZE, IMAGE_SIZE), antialias=True)
    img = TF.normalize(img, mean=IMG_MEAN, std=IMG_STD)
    nbox = torch.tensor([[sbox[0] / W, sbox[1] / H, sbox[2] / W, sbox[3] / H]], dtype=torch.float32)
    if clamp_box:
        nbox = nbox.clamp(0.0, 1.0)  # src/datasets/{childplay,videoattentiontarget,gazefollow}.py (test loaders)
    sample = {
        "image": img.unsqueeze(0).to(device),  # (1, 3, 224, 224)
        "heads": heads.unsqueeze(0).to(device),  # (1, n=1, 3, 224, 224)
        "head_bboxes": nbox.unsqueeze(0).to(device),  # (1, 1, 4)
    }
    return sample, sbox


def predict(model, image, box, device: str, expand: float = 0.0, clamp_box: bool = True):
    sample, sbox = build_inputs(image, box, device, expand, clamp_box)
    with torch.no_grad():
        _gaze_vec, heatmaps, inout = model(sample)  # (1, n, 2), (1, n, 64, 64), (1, n, 1)  sharingan.py L537-564
    heatmap = heatmaps[0, -1].detach().float().cpu().numpy()  # target = last person; raw 64x64
    p_in = sigmoid(inout[0, -1, 0].detach().float().cpu())  # logits -> sigmoid (demo.py L194)
    return heatmap, p_in, sbox


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="childplay", choices=sorted(VARIANTS))
    ap.add_argument("--checkpoint", default=None, help="defaults to the variant's checkpoint")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--debug-dir", default=None, help="write overlays of the first rows here")
    ap.add_argument("--expand-box", type=float, default=0.0,
                    help="expand the GT box by k*size per side before squaring (0 = repo ChildPlay loader; VAT uses 0.05, GazeFollow 0.1)")
    ap.add_argument("--no-clamp-box", action="store_true", help="do not clamp the normalised box to [0,1] (demo.py behaviour)")
    args = ap.parse_args()
    checkpoint = args.checkpoint or VARIANTS[args.variant][0]

    frames = load_frame_list(args.frame_list)
    model = load_model(checkpoint, args.device, args.repo)
    meta = {
        "model": "sharingan", "variant": args.variant, "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint), "repo": args.repo, "model_args": MODEL_ARGS,
        "training_data": VARIANTS[args.variant][1],
        "preprocessing": {
            "box": f"ground-truth head box, expand={args.expand_box}, squared around its centre (side = max(w,h), repo "
                   f"square_bbox, not clipped); normalised xyxy {'not clamped' if args.no_clamp_box else 'clamped to [0,1]'}",
            "head": "crop of the squared box (PIL crop, black outside the frame) -> 224x224 bilinear antialias, repo mean/std",
            "image": "224x224 squash, bilinear antialias, repo mean/std",
            "people": "target person only (single gaze token = last token)",
            "heatmap": "raw decoder output, 64x64 (no sigmoid)", "inout": "sigmoid of the in/out logit"},
        "has_inout": True, "end_to_end": False,
        "frame_list": os.path.abspath(args.frame_list), "frame_list_sha256": sha256_file(args.frame_list),
        **environment_meta(),
    }
    writer = CacheWriter(args.out, frames, meta)
    if args.resume:
        print(f"[sharingan] resuming: {writer.resume()} rows already done")

    t0, n, debug_left = time.time(), 0, 4 if args.debug_dir else 0
    total = frames.iloc[: args.limit]["image_path"].nunique() if args.limit else frames["image_path"].nunique()
    for image, rows in tqdm(iter_frames(frames, limit=args.limit), total=total, desc=f"sharingan-{args.variant}"):
        for _, row in rows.iterrows():
            i = int(row["row"])
            if writer.done[i]:
                continue
            box = head_xyxy(row)
            heatmap, inout, sbox = predict(model, image, box, args.device, args.expand_box, not args.no_clamp_box)
            writer.add(i, heatmap, inout, model_box=sbox)
            n += 1
            if debug_left > 0:
                overlay_debug_png(image, sbox, heatmap, writer.point_norm[i], (row["gaze_x"], row["gaze_y"]),
                                  os.path.join(args.debug_dir, f"sharingan_{args.variant}_{i:05d}.png"),
                                  title=f"{row['clip']} p{row['person_id']} f{row['frame']} inout={inout:.2f} gt={row['gaze_class']}")
                debug_left -= 1
    writer.save()
    print(f"[sharingan] {n} predictions in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
