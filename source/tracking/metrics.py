"""Gaze-following metrics with the VideoAttentionTarget protocol: AUC on the
64x64 heatmap, L2 distance in normalized coordinates, and in/out AP.

AUC needs a heatmap, so for the tracking agent it is computed on trigger
frames only. L2 is computed on every frame.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def is_evaluable(gaze_class) -> bool:
    return not pd.isna(gaze_class) and gaze_class == "inside_visible"


def inout_label(gaze_class):
    """1 = inside the frame, 0 = outside, None = occluded/uncertain (ignored)."""
    if pd.isna(gaze_class):
        return None
    return {"inside_visible": 1, "outside_frame": 0}.get(gaze_class)


def auc(heatmap: torch.Tensor, gt_x: float, gt_y: float, res: int = 64, sigma: int = 3) -> float:
    if heatmap.shape[-2:] != (res, res):
        heatmap = torch.nn.functional.interpolate(
            heatmap.float().reshape(1, 1, *heatmap.shape[-2:]), size=(res, res), mode="bilinear"
        ).reshape(res, res)
    target = np.zeros((res, res))
    gx, gy = gt_x * res, gt_y * res
    ul = [max(0, int(gx - 3 * sigma)), max(0, int(gy - 3 * sigma))]
    br = [min(int(gx + 3 * sigma + 1), res - 1), min(int(gy + 3 * sigma + 1), res - 1)]
    target[ul[1]:br[1], ul[0]:br[0]] = 1
    return float(roc_auc_score(target.flatten(), heatmap.cpu().flatten()))


def l2(pred_x: float, pred_y: float, gt_x: float, gt_y: float) -> float:
    return float(np.hypot(pred_x - gt_x, pred_y - gt_y))


def score_frame(row, image_size, point, heatmap) -> dict:
    out = {
        "gt_evaluable": is_evaluable(row.get("gaze_class")),
        "inout_gt": inout_label(row.get("gaze_class")),
        "auc": float("nan"),
        "l2": float("nan"),
    }
    if out["gt_evaluable"] and not np.isnan(point[0]):
        W, H = image_size
        gt_x, gt_y = row["gaze_x"] / W, row["gaze_y"] / H
        out["l2"] = l2(point[0] / W, point[1] / H, gt_x, gt_y)
        if heatmap is not None:
            out["auc"] = auc(heatmap, gt_x, gt_y)
    return out


def summarize(df: pd.DataFrame) -> dict:
    evaluable = df[df["gt_evaluable"]]
    auc_rows = evaluable[evaluable["auc"].notna()]
    mask = df["inout_gt"].notna() & df["inout_score"].notna()
    labels = df.loc[mask, "inout_gt"].astype(int)
    inout_ap = (
        float(average_precision_score(labels, df.loc[mask, "inout_score"]))
        if mask.sum() >= 2 and labels.nunique() > 1 else float("nan")
    )
    return {
        "frames": int(len(df)),
        "estimator_calls": int(df["is_trigger_frame"].sum()),
        "auc": float(auc_rows["auc"].mean()) if len(auc_rows) else float("nan"),
        "l2": float(evaluable["l2"].mean()) if len(evaluable) else float("nan"),
        "inout_ap": inout_ap,
    }
