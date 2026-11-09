#!/usr/bin/env python3
"""Stage 2: GTA gaze-guided grounding on the tracking agent's output ("Ours w/ PaGE").

For every person-frame with a gaze estimate, the candidate boxes (detector score >= 0.3, as in GTA's gazehoi.yaml) are
reduced to one gaze-target box with GTA's own rule (Gaze-Target-Agent source/models/grounding.py, copied here):
  * estimator frame (trigger, in-frame; the PaGE heatmap exists): among the boxes that contain the gaze point take the one
    with the highest "sqrt_norm_heatmap" = heatmap mass in the box / sqrt(box area);
  * frames in between (the agent only follows the point with optical flow, no heatmap): the smallest box containing the point;
  * no box contains the point: the box whose centre is nearest (GTA's fallback).
Skipped: estimator frames judged out-of-frame (nothing to ground) and frames with no gaze point (track lost after an out-of-frame trigger).
"""
import argparse, json, math, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np, pandas as pd, torch

THRESH = 0.3


def contains(b, p): return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
def area(b): return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
def centre_dist(b, p): return math.hypot((b[0] + b[2]) / 2 - p[0], (b[1] + b[3]) / 2 - p[1])


def sqrt_norm(hm_full, b, W, H):
    x1, y1, x2, y2 = max(0, int(round(b[0]))), max(0, int(round(b[1]))), min(W, int(round(b[2]))), min(H, int(round(b[3])))
    if x2 <= x1 or y2 <= y1: return 0.0
    r = hm_full[y1:y2, x1:x2]
    return float(r.sum()) / math.sqrt(r.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--tracks", default="outputs/tracking/page_tracking.csv")
    ap.add_argument("--cache", default="outputs/gaze_benchmark/caches/page.npz")
    ap.add_argument("--out", default="outputs/grounding")
    a = ap.parse_args()
    d = pd.read_csv(a.tracks, dtype={"person_id": str})
    z = np.load(a.cache, allow_pickle=False)
    idx = {(c, p, int(f)): i for i, (c, p, f) in enumerate(zip(z["clip"].astype(str), z["person_id"].astype(str), z["frame"].astype(int)))}
    hm64 = z["heatmap64"]
    dets = {}
    for line in open(os.path.join(a.out, a.detector, "detections.jsonl")):
        r = json.loads(line)
        keep = [i for i, s in enumerate(r["scores"]) if s >= THRESH]
        dets[(r["clip"], r["frame"])] = (r["w"], r["h"], [r["boxes"][i] for i in keep], [r["scores"][i] for i in keep], [r["classes"][i] for i in keep])
    rows = []
    for t in d.itertuples(index=False):
        base = {"clip": t.clip, "person_id": t.person_id, "frame": t.frame, "is_trigger_frame": bool(t.is_trigger_frame)}
        if t.is_trigger_frame and not t.in_frame:
            rows.append({**base, "mode": "skipped_out_of_frame"}); continue
        if np.isnan(t.point_x):
            rows.append({**base, "mode": "skipped_no_gaze_point"}); continue
        key = (t.clip, int(t.frame))
        if key not in dets:
            rows.append({**base, "mode": "skipped_no_detections_row"}); continue
        W, H, boxes, scores, classes = dets[key]
        p = (float(t.point_x), float(t.point_y))
        row = {**base, "gaze_x": p[0], "gaze_y": p[1], "n_candidates": len(boxes)}
        if not boxes:
            rows.append({**row, "mode": "no_candidates"}); continue
        inside = [i for i, b in enumerate(boxes) if contains(b, p)]
        row["n_containing"] = len(inside)
        use_heatmap = bool(t.is_trigger_frame)
        if inside:
            if use_heatmap:
                hm = torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(hm64[idx[(t.clip, t.person_id, int(t.frame))]]))[None, None],
                                                    size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()
                sc = [(sqrt_norm(hm, boxes[i], W, H), i) for i in inside]
                best = max(sc)[1]; row.update(mode="sqrt_norm_heatmap", select_score=max(sc)[0])
            else:
                best = min(inside, key=lambda i: area(boxes[i])); row.update(mode="gaze_point")
            row["fallback"] = False
        else:
            best = min(range(len(boxes)), key=lambda i: centre_dist(boxes[i], p))
            row.update(mode="sqrt_norm_heatmap" if use_heatmap else "gaze_point", fallback=True)
        b = boxes[best]
        row.update(x1=b[0], y1=b[1], x2=b[2], y2=b[3], det_score=scores[best], det_class=classes[best], box_area_frac=area(b) / (W * H))
        rows.append(row)
    out = pd.DataFrame(rows)
    path = os.path.join(a.out, a.detector, "gaze_target_boxes.csv"); out.to_csv(path, index=False)
    g = out[out["x1"].notna()]
    summ = {"detector": a.detector, "person_frames": len(out), "grounded": int(len(g)), "mode_counts": out["mode"].value_counts().to_dict(),
            "fallback_nearest_centre_pct": round(100 * float(g["fallback"].mean()), 2), "mean_candidates": round(float(g["n_candidates"].mean()), 2),
            "mean_containing": round(float(g["n_containing"].mean()), 2), "median_box_area_frac": round(float(g["box_area_frac"].median()), 4)}
    json.dump(summ, open(os.path.join(a.out, a.detector, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1)); print("wrote", path)


if __name__ == "__main__":
    main()
