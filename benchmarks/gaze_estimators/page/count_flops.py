#!/usr/bin/env python3
"""FLOPs and latency of one PaGE call (scene + one head crop + one box)."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image

from benchmarks.common import count_flops, environment_meta, head_xyxy, load_frame_list, time_call, write_json
from export_dense import build_inputs, load_model


def shapes(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [shapes(v) for v in obj]
    if isinstance(obj, dict):
        return {k: shapes(v) for k, v in obj.items()}
    return str(obj)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="Octopus1/page-vitb")
    ap.add_argument("--variant", default="vitb")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    row = load_frame_list(args.frame_list).iloc[0]
    image = Image.open(row["image_path"]).convert("RGB")
    model, processor = load_model(args.model_path, args.device)
    inputs, _ = build_inputs(processor, image, head_xyxy(row), args.device)
    res = count_flops(model, inputs)
    res["cuda_ms_per_call"] = time_call(model, inputs)
    res.update(model="page", variant=args.variant, checkpoint=args.model_path, input_shapes=shapes(inputs),
               aux={}, aux_flops=0, total_flops=res["flops"], total_tflops=res["tflops"], **environment_meta())
    write_json(args.out, res)
    print(f"[flops] page-{args.variant} {res['tflops']:.4f} TFLOPs/call, {res['cuda_ms_per_call']:.1f} ms/call -> {args.out}")


if __name__ == "__main__":
    main()
