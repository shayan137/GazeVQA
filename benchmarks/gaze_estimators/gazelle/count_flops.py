#!/usr/bin/env python3
"""FLOPs and latency of one Gaze-LLE call (one 448x448 frame, one head box)."""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from PIL import Image

from benchmarks.common import count_flops, environment_meta, head_xyxy, load_frame_list, time_call, write_json
from source.tracking.estimators import GazeLLEEstimator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="fkryan/gazelle")
    ap.add_argument("--model", default="gazelle_dinov2_vitl14_inout")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.environ.setdefault("TORCH_HOME", os.path.join("model_cache", "torch_hub"))
    row = load_frame_list(args.frame_list).iloc[0]
    image = Image.open(row["image_path"]).convert("RGB")
    W, H = image.size
    x1, y1, x2, y2 = head_xyxy(row)
    est = GazeLLEEstimator(args.repo, args.model, device=args.device)
    inp = {
        "images": est.transform(image).unsqueeze(0).to(args.device),
        "bboxes": [[(max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H))]],
    }
    res = count_flops(est.model, inp)
    res["cuda_ms_per_call"] = time_call(est.model, inp)
    res.update(
        model="gazelle", variant=args.model, input_shapes={"images": list(inp["images"].shape)},
        aux={}, aux_flops=0, total_flops=res["flops"], total_tflops=res["tflops"], **environment_meta(),
    )
    write_json(args.out, res)
    print(f"[flops] gazelle {res['tflops']:.4f} TFLOPs/call, {res['cuda_ms_per_call']:.1f} ms/call -> {args.out}")


if __name__ == "__main__":
    main()
