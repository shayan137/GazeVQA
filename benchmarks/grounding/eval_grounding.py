#!/usr/bin/env python3
"""
Score the gaze-target grounding runs (outputs/grounding/<detector>/gaze_target_boxes.csv, made by select_gaze_target.py from the gaze
of the tracking agent) against the annotated boxes of the attended objects in the test split.

  python benchmarks/grounding/eval_grounding.py --ann-dir <annotation folder with the test CSVs> [--runs outputs/grounding]

The evaluation set is every test person-frame that has an annotated object box (obj_bbox_* in <clip>.csv) and is a row of the tracking
output. A person-frame without a predicted box (no gaze point, or gaze outside the frame) counts as an IoU of 0.
Metrics: mIoU, Acc@50 and Acc@75 (share of person-frames with an IoU of at least 0.5 and 0.75) and AP@50 (VOC all-point average precision at
an IoU of 0.5, one box per person-frame ranked by the detector score, recall over all evaluated person-frames). The mIoU is also split by
the source of the annotated box and by the kind of target (person or object).
The results are written to <runs>/grounding_eval.csv.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

DETECTORS = [("rfdetr_2xl", "RF-DETR 2XL"), ("yolo26x", "YOLO26x"), ("wedetect_large_uni", "WeDetect-Large-Uni")]
PERSON = r"\b(?:man|woman|boy|girl|baby|child|kid|lady|person|adult|toddler|mother|father)\b"

def load_gt(ann_dir):
    files = [f for f in glob.glob(os.path.join(ann_dir, "*.csv")) if not f.endswith(("_VQA.csv", "_temporal.csv")) and "temporal_qa" not in f]
    g = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    g = g[g["obj_bbox_x"] != ""].copy()
    for c in ("obj_bbox_x", "obj_bbox_y", "obj_bbox_width", "obj_bbox_height"):
        g[c] = g[c].astype(float)
    g["gx1"], g["gy1"] = g["obj_bbox_x"], g["obj_bbox_y"]
    g["gx2"], g["gy2"] = g["gx1"] + g["obj_bbox_width"], g["gy1"] + g["obj_bbox_height"]
    g["frame"] = g["frame"].astype(int)
    g["person_id"] = g["person_id"].astype(str)
    g["target"] = np.where(g["objects"].str.split(",").str[0].str.lower().str.contains(PERSON, regex=True), "person", "object")
    return len(files), g[["clip", "frame", "person_id", "gx1", "gy1", "gx2", "gy2", "obj_bbox_source", "target"]]


def iou(a, b):
    ix = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = ix * iy
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.nan_to_num(inter / np.maximum(union, 1e-9))


def ap50(conf, hit, n_gt):
    """VOC all-point average precision: predictions ranked by conf, hit = IoU >= 0.5 (one prediction per ground-truth box)."""
    order = np.argsort(-conf, kind="stable")
    tp = hit[order].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0.0], rec, [1.0]]), np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--runs", default="outputs/grounding")
    a = ap.parse_args()
    n_clips, gt = load_gt(a.ann_dir)
    print(f"{n_clips} test clips, {len(gt)} person-frames with an annotated object box; sources {gt['obj_bbox_source'].value_counts().to_dict()}")
    csv_rows = []
    for det, name in DETECTORS:
        p = pd.read_csv(os.path.join(a.runs, det, "gaze_target_boxes.csv"), dtype={"person_id": str})
        ev = gt.merge(p, on=["clip", "person_id", "frame"], how="inner")
        v = iou(ev[["x1", "y1", "x2", "y2"]].to_numpy(float), ev[["gx1", "gy1", "gx2", "gy2"]].to_numpy(float))
        ev["iou"] = v
        on = ev["x1"].notna().to_numpy()
        r = {"detector": det, "name": name, "n_eval": len(ev), "n_grounded": int(on.sum()), "mIoU": 100 * v.mean(),
             "Acc@50": 100 * (v >= .5).mean(), "Acc@75": 100 * (v >= .75).mean(),
             "AP@50": 100 * ap50(ev["det_score"].to_numpy(float)[on], (v >= .5)[on], len(ev))}
        for k, sub in ev.groupby("target"):
            r[f"mIoU[{k}]"], r[f"n[{k}]"] = 100 * sub["iou"].mean(), len(sub)
        for k, sub in ev.groupby("obj_bbox_source"):
            r[f"mIoU[{k}]"] = 100 * sub["iou"].mean()
        csv_rows.append(r)
        print(f"{name:20s} n={r['n_eval']} grounded={r['n_grounded']} mIoU {r['mIoU']:.1f}  Acc@50 {r['Acc@50']:.1f}  Acc@75 {r['Acc@75']:.1f}  AP@50 {r['AP@50']:.1f}  "
              f"person targets {r['mIoU[person]']:.1f} (n={r['n[person]']})  other objects {r['mIoU[object]']:.1f} (n={r['n[object]']})")
    pd.DataFrame(csv_rows).round(2).to_csv(os.path.join(a.runs, "grounding_eval.csv"), index=False)
    print("wrote", os.path.join(a.runs, "grounding_eval.csv"))


if __name__ == "__main__":
    main()
