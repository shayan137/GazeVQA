"""Shared helpers for the third-party gaze-estimator benchmark.

Every estimator is run once, densely, over the benchmark's person-frames in
its own Python environment (benchmarks/gaze_estimators/<model>/export_dense.py)
and writes a prediction cache; scripts/run_tracking.py --estimator cached then
replays that cache in dense and in tracking mode. This module depends only on
numpy, pandas, torch and Pillow so that every environment can import it.

Cache file (.npz), one row per frame-list row in frame-list order:
  clip, person_id     str
  frame               int32
  image_wh            int32   (N, 2)      image width, height in pixels
  head_xyxy           float32 (N, 4)      ground-truth head box given to the model
  model_box_xyxy      float32 (N, 4)      head box after the model's own preprocessing
  heatmap64           float32 (N, 64, 64) native heatmap resized to 64x64 (bilinear, as metrics.auc does)
  native_hw           int32   (N, 2)      native heatmap height, width
  point_norm          float64 (N, 2)      argmax of the native heatmap, cell centre, in [0, 1]
  inout               float64 (N,)        P(gaze inside the frame); NaN if the model has no in/out head
  ok                  bool    (N,)        False if the model produced no prediction for the row
  done                bool    (N,)        row has been processed (for --resume)
  match_iou           float32 (N,)        end-to-end models: IoU of the matched predicted head with the GT head
  meta                str                 JSON: model, variant, checkpoint, preprocessing, versions, ...
"""

import contextlib
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

FRAME_LIST_COLUMNS = [
    "clip", "person_id", "frame", "image_path", "W", "H",
    "head_x1", "head_y1", "head_x2", "head_y2", "gaze_class", "gaze_x", "gaze_y", "look_like",
]
HEATMAP_RES = 64
CACHE_ARRAYS = [
    "clip", "person_id", "frame", "image_wh", "head_xyxy", "model_box_xyxy", "heatmap64",
    "native_hw", "point_norm", "inout", "ok", "done", "match_iou",
]


def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def environment_meta() -> dict:
    meta = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": platform.node(),
    }
    try:
        import torchvision
        meta["torchvision"] = torchvision.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
    return meta


def load_frame_list(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"clip": str, "person_id": str, "look_like": str})
    missing = [c for c in FRAME_LIST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"frame list {path} lacks columns {missing}")
    df["frame"] = df["frame"].astype(int)
    df = df.reset_index(drop=True)
    df["row"] = np.arange(len(df))
    return df


def iter_frames(df: pd.DataFrame, limit: int | None = None):
    """Yields (PIL RGB image, rows of that image) once per unique image, in
    order of first appearance, so each frame is decoded once."""
    if limit is not None:
        df = df.iloc[:limit]
    for path, rows in df.groupby("image_path", sort=False):
        with Image.open(path) as im:
            image = im.convert("RGB")
        yield image, rows


def head_xyxy(row) -> tuple[float, float, float, float]:
    return float(row["head_x1"]), float(row["head_y1"]), float(row["head_x2"]), float(row["head_y2"])


def expand_box(box, k: float, W: int, H: int) -> tuple[float, float, float, float]:
    """Enlarges a box by k * its size on every side and clips it to the image."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return max(0.0, x1 - k * w), max(0.0, y1 - k * h), min(float(W), x2 + k * w), min(float(H), y2 + k * h)


def square_box(box, W: int, H: int) -> tuple[float, float, float, float]:
    """Makes a box square around its centre (side = longer side) and clips it."""
    x1, y1, x2, y2 = box
    cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
    return max(0.0, cx - side / 2), max(0.0, cy - side / 2), min(float(W), cx + side / 2), min(float(H), cy + side / 2)


def to_numpy_heatmap(hm) -> np.ndarray:
    if isinstance(hm, torch.Tensor):
        hm = hm.detach().float().cpu().numpy()
    hm = np.squeeze(np.asarray(hm, dtype=np.float32))
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2-D after squeezing, got shape {hm.shape}")
    return hm


def heatmap_to_64(hm, res: int = HEATMAP_RES) -> np.ndarray:
    hm = to_numpy_heatmap(hm)
    if hm.shape == (res, res):
        return hm
    t = torch.from_numpy(np.ascontiguousarray(hm)).reshape(1, 1, *hm.shape)
    return F.interpolate(t, size=(res, res), mode="bilinear").reshape(res, res).numpy()


def argmax_point_norm(hm) -> tuple[float, float]:
    """Centre of the argmax cell in normalised image coordinates, the same
    convention as GazeLLEEstimator."""
    hm = to_numpy_heatmap(hm)
    h, w = hm.shape
    idx = int(np.argmax(hm.reshape(-1)))
    return (idx % w + 0.5) / w, (idx // w + 0.5) / h


def sigmoid(x) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def box_iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / area) if area > 0 else 0.0


class CacheWriter:
    """Accumulates per-row predictions and writes the cache file atomically."""

    def __init__(self, path: str, frame_list: pd.DataFrame, meta: dict, flush_every: int = 500):
        n = len(frame_list)
        self.path, self.n, self.meta, self.flush_every = path, n, dict(meta), flush_every
        # fixed-width unicode arrays, so the cache loads with allow_pickle=False
        self.clip = np.array(frame_list["clip"].astype(str).tolist(), dtype=str)
        self.person_id = np.array(frame_list["person_id"].astype(str).tolist(), dtype=str)
        self.frame = frame_list["frame"].to_numpy(np.int32)
        self.image_wh = frame_list[["W", "H"]].to_numpy(np.int32)
        self.head_xyxy = frame_list[["head_x1", "head_y1", "head_x2", "head_y2"]].to_numpy(np.float32)
        self.model_box_xyxy = np.full((n, 4), np.nan, np.float32)
        self.heatmap64 = np.zeros((n, HEATMAP_RES, HEATMAP_RES), np.float32)
        self.native_hw = np.zeros((n, 2), np.int32)
        self.point_norm = np.full((n, 2), np.nan, np.float64)
        self.inout = np.full(n, np.nan, np.float64)
        self.ok = np.zeros(n, bool)
        self.done = np.zeros(n, bool)
        self.match_iou = np.full(n, np.nan, np.float32)
        self._since_flush = 0

    def resume(self) -> int:
        """Loads rows already written to self.path (same frame list). Returns how many."""
        if not os.path.isfile(self.path):
            return 0
        z = np.load(self.path, allow_pickle=False)
        if len(z["frame"]) != self.n or not (z["clip"].astype(str) == self.clip).all() \
                or not (z["person_id"].astype(str) == self.person_id).all() or not (z["frame"] == self.frame).all():
            raise ValueError(f"{self.path} was written for a different frame list")
        for k in ("model_box_xyxy", "heatmap64", "native_hw", "point_norm", "inout", "ok", "match_iou"):
            setattr(self, k, z[k].copy())
        self.done = z["done"].copy() if "done" in z.files else z["ok"].copy()
        return int(self.done.sum())

    def add(self, i: int, heatmap_native, inout, model_box=None, match_iou=float("nan"), ok: bool = True):
        if ok:
            hm = to_numpy_heatmap(heatmap_native)
            self.native_hw[i] = hm.shape
            self.heatmap64[i] = heatmap_to_64(hm)
            self.point_norm[i] = argmax_point_norm(hm)
            self.inout[i] = float("nan") if inout is None else float(inout)
        if model_box is not None:
            self.model_box_xyxy[i] = np.asarray(model_box, np.float32)
        self.match_iou[i] = match_iou
        self.ok[i] = bool(ok)
        self.done[i] = True
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.save()

    def save(self):
        meta = dict(self.meta)
        meta.update(
            rows=int(self.n), rows_done=int(self.done.sum()), rows_ok=int(self.ok.sum()),
            partial=bool(not self.done.all()), saved_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, **{k: getattr(self, k) for k in CACHE_ARRAYS}, meta=np.array(json.dumps(meta)))
        os.replace(tmp, self.path)
        self._since_flush = 0


def read_cache(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(z["meta"].item()) if "meta" in z.files else {}
    return out


def count_flops(fn, *args, **kwargs) -> dict:
    """FLOPs of one call of fn (2 FLOPs per multiply-add, torch's
    FlopCounterMode). Fused attention kernels are disabled for the count so
    that attention is decomposed into matmuls and gets counted."""
    from torch.utils.flop_counter import FlopCounterMode

    fast = None
    mha = getattr(torch.backends, "mha", None)
    if mha is not None and hasattr(mha, "get_fastpath_enabled"):
        fast = mha.get_fastpath_enabled()
        mha.set_fastpath_enabled(False)
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel(SDPBackend.MATH)
    except Exception:
        ctx = contextlib.nullcontext()
    try:
        with torch.no_grad(), ctx, FlopCounterMode(display=False) as fc:
            fn(*args, **kwargs)
    finally:
        if fast is not None:
            mha.set_fastpath_enabled(fast)
    total = int(fc.get_total_flops())
    counts = fc.get_flop_counts()
    by_op = {str(op): int(n) for op, n in counts.get("Global", {}).items()}
    by_module = {name: int(sum(ops.values())) for name, ops in counts.items() if name != "Global"}
    return {"flops": total, "tflops": total / 1e12, "by_op": by_op, "by_module": by_module}


def time_call(fn, *args, n: int = 50, warmup: int = 5, **kwargs) -> float:
    """Mean wall-clock milliseconds per call, CUDA-synchronised."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            fn(*args, **kwargs)
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn(*args, **kwargs)
        sync()
    return (time.perf_counter() - t0) / n * 1000.0


def write_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


def overlay_debug_png(image: Image.Image, head_box, heatmap, point_norm, gt_xy_px, out_path: str, title: str = ""):
    """Heatmap (red), head box (yellow), predicted point (green) and GT point
    (blue) drawn on the frame, for eyeballing a new adapter."""
    W, H = image.size
    hm = to_numpy_heatmap(heatmap)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    heat = np.asarray(Image.fromarray((hm * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), np.float32) / 255.0
    base = np.asarray(image, np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    blend = base * (1 - 0.55 * heat[..., None]) + red * (0.55 * heat[..., None])
    canvas = Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(canvas)
    if head_box is not None:
        d.rectangle([head_box[0], head_box[1], head_box[2], head_box[3]], outline=(255, 255, 0), width=4)
    if point_norm is not None and np.isfinite(point_norm[0]):
        px, py = point_norm[0] * W, point_norm[1] * H
        d.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(0, 255, 0), width=5)
    if gt_xy_px is not None and np.isfinite(gt_xy_px[0]) and gt_xy_px[0] >= 0:
        gx, gy = gt_xy_px
        d.ellipse([gx - 12, gy - 12, gx + 12, gy + 12], outline=(0, 128, 255), width=5)
    if title:
        d.text((10, 10), title, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    canvas.save(out_path)
