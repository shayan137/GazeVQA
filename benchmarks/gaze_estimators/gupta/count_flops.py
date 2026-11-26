#!/usr/bin/env python3
"""FLOPs and latency of one Gupta et al. (CVPRW 2022) call (one frame, one
head) plus the two auxiliary modalities it needs: MiDaS DPT_Large depth
(flops/aux_midas_dpt_large.json) and KeypointRCNN pose
(flops/aux_pose_keypointrcnn.json)."""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from benchmarks.common import count_flops, environment_meta, head_xyxy, load_frame_list, time_call, write_json
from export_dense import CKPTS
from model_api import aux_paths, build, build_inputs, forward, load_depth, load_pose


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="vat", choices=list(CKPTS))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--aux-dir", default=os.path.join(ROOT, "outputs", "gaze_benchmark", "aux"))
    ap.add_argument("--aux-flops-json", nargs="+",
                    default=[os.path.join(ROOT, "outputs", "gaze_benchmark", "flops", "aux_midas_dpt_large.json"),
                             os.path.join(ROOT, "outputs", "gaze_benchmark", "flops", "aux_pose_keypointrcnn.json")])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    ckpt = args.checkpoint or CKPTS[args.variant]

    row = load_frame_list(args.frame_list).iloc[0]
    image = Image.open(row["image_path"]).convert("RGB")
    dp, pp = aux_paths(args.aux_dir, row["image_path"])
    model = build(ckpt, args.device)
    *inputs, _ = build_inputs(image, load_depth(dp), load_pose(pp), head_xyxy(row), args.device)
    fn = lambda *t: forward(model, *t)  # noqa: E731
    res = count_flops(fn, *inputs)
    res["cuda_ms_per_call"] = time_call(fn, *inputs)
    aux, aux_ms = {}, {}
    for p in args.aux_flops_json:
        with open(p) as f:
            j = json.load(f)
        aux[j["model"]] = j["flops"]
        aux_ms[j["model"]] = j.get("cuda_ms_per_call")
    aux_flops = sum(aux.values())
    names = ["image", "depth", "pose", "face", "gaze_field", "head"]
    res.update(model="gupta", variant=args.variant, checkpoint=ckpt,
               model_args={"cone_mode": model.cone_mode_used, "pred_inout": model.pred_inout_used},
               input_shapes={k: list(t.shape) for k, t in zip(names, inputs)},
               aux=aux, aux_flops=aux_flops, aux_cuda_ms=aux_ms,
               total_flops=res["flops"] + aux_flops, total_tflops=(res["flops"] + aux_flops) / 1e12, **environment_meta())
    write_json(args.out, res)
    print(f"[flops] gupta-{args.variant} {res['tflops']:.4f} T model + {aux_flops / 1e12:.4f} T aux {list(aux)} = {res['total_tflops']:.4f} TFLOPs/call, {res['cuda_ms_per_call']:.1f} ms/call -> {args.out}")


if __name__ == "__main__":
    main()
