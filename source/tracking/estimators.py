"""Image-based gaze estimators: frame + person -> gaze point, in/out score
and heatmap. Gaze-LLE is prompted with the head box, GazeAnywhere with the
person's appearance description. CachedEstimator replays a dense prediction
cache written by benchmarks/gaze_estimators/<model>/export_dense.py, which is
how third-party estimators are evaluated in both tracking modes.
"""

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image


@dataclass
class PersonContext:
    head_bbox_px: tuple[float, float, float, float] | None = None
    description: str | None = None
    # identity of the person-frame, needed by CachedEstimator
    clip: str | None = None
    person_id: str | None = None
    frame: int | None = None


@dataclass
class GazePrediction:
    x_px: float
    y_px: float
    inout_score: float
    heatmap: torch.Tensor | None  # on CPU, used for AUC


class GazeLLEEstimator:
    def __init__(self, repo: str = "fkryan/gazelle", model_name: str = "gazelle_dinov2_vitl14_inout", device: str = "cuda"):
        self.device = device
        self.model, self.transform = torch.hub.load(repo, model_name, trust_repo=True)
        self.model.eval().requires_grad_(False).to(device)

    def predict(self, image: Image.Image, person: PersonContext) -> GazePrediction:
        if person.head_bbox_px is None:
            raise ValueError("Gaze-LLE needs a head bounding box")
        W, H = image.size
        x1, y1, x2, y2 = person.head_bbox_px
        bbox_norm = (max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H))
        inp = {
            "images": self.transform(image).unsqueeze(0).to(self.device),
            "bboxes": [[bbox_norm]],
        }
        with torch.no_grad():
            out = self.model(inp)

        heatmap = out["heatmap"][0][0]
        inout = float(out["inout"][0][0]) if out.get("inout") is not None else 1.0
        Hh, Wh = heatmap.shape[-2], heatmap.shape[-1]
        idx = int(torch.argmax(heatmap.reshape(-1)).item())
        gx = (idx % Wh + 0.5) / Wh
        gy = (idx // Wh + 0.5) / Hh
        return GazePrediction(x_px=gx * W, y_px=gy * H, inout_score=inout, heatmap=heatmap.detach().cpu())


class GazeAnywhereEstimator:
    def __init__(self, model_id: str = "IrohXu/GazeAnywhere", repo_dir: str = "", device: str = "cuda"):
        # The HF remote code imports `modeling` from the GitHub repo, which is
        # not part of the HF release.
        if repo_dir and repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
        # gaze_model.device is a plain attribute fixed to "cpu" upstream and not
        # updated by .to(), so text tokens would stay on CPU.
        self.model.gaze_model.device = torch.device(device)

    def predict(self, image: Image.Image, person: PersonContext) -> GazePrediction:
        text = person.description
        if pd.isna(text) or not text:
            text = "the person"
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_gaze_estimation(outputs, target_sizes=inputs["original_sizes"].tolist())[0]

        gx, gy = result["gaze"]
        inout = result["inout"]
        heatmap = result.get("heatmap")
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
        elif heatmap is not None:
            heatmap = heatmap.detach().cpu()
        if isinstance(inout, bool):
            inout = 1.0 if inout else 0.0
        return GazePrediction(x_px=float(gx), y_px=float(gy), inout_score=float(inout), heatmap=heatmap)


class CachedEstimator:
    """Replays per-person-frame predictions from a dense cache (.npz).

    The cache is written by benchmarks/gaze_estimators/<model>/export_dense.py
    (file contract in benchmarks/common.py). Because every estimator is a
    deterministic function of (frame, head box), replaying its dense
    predictions on the trigger frames gives exactly what calling it live
    would; the dense and tracking runs then differ only in which frames count
    as estimator calls. A frame missing from the cache is an error.
    """

    needs_pixels = False  # predictions come from the cache, the image is only used for its size

    def __init__(self, cache_path: str):
        if not cache_path:
            raise ValueError("estimator 'cached' needs cache_path (--cache)")
        z = np.load(cache_path, allow_pickle=False)
        clips = z["clip"].astype(str)
        pids = z["person_id"].astype(str)
        frames = z["frame"].astype(int)
        self.index = {(c, p, int(f)): i for i, (c, p, f) in enumerate(zip(clips, pids, frames))}
        if len(self.index) != len(clips):
            raise ValueError(f"duplicate (clip, person_id, frame) rows in {cache_path}")
        self.heatmap = z["heatmap64"]
        self.point = z["point_norm"]
        self.inout = z["inout"]
        self.ok = z["ok"]
        self.meta = json.loads(z["meta"].item()) if "meta" in z.files else {}
        self.cache_path = cache_path
        if self.meta.get("partial"):
            print(f"[cached] warning: {cache_path} is a partial cache ({self.meta.get('rows_done')}/{self.meta.get('rows')} rows)")

    def predict(self, image: Image.Image, person: PersonContext) -> GazePrediction:
        key = (str(person.clip), str(person.person_id), int(person.frame))
        try:
            i = self.index[key]
        except KeyError:
            raise KeyError(f"{key} is not in the prediction cache {self.cache_path}") from None
        if not bool(self.ok[i]):
            return GazePrediction(float("nan"), float("nan"), float("nan"), None)
        W, H = image.size
        gx, gy = float(self.point[i, 0]), float(self.point[i, 1])
        heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap[i], dtype=np.float32))
        return GazePrediction(x_px=gx * W, y_px=gy * H, inout_score=float(self.inout[i]), heatmap=heatmap)


def build_estimator(cfg):
    os.environ.setdefault("TORCH_HOME", os.path.join(cfg.model_cache_dir, "torch_hub"))
    os.environ.setdefault("HF_HOME", os.path.join(cfg.model_cache_dir, "huggingface"))
    if cfg.estimator == "gazelle":
        return GazeLLEEstimator(cfg.gazelle_repo, cfg.gazelle_model, device=cfg.device)
    if cfg.estimator == "gaze_anywhere":
        return GazeAnywhereEstimator(cfg.gaze_anywhere_model, cfg.gaze_anywhere_repo, device=cfg.device)
    if cfg.estimator == "cached":
        return CachedEstimator(cfg.cache_path)
    raise ValueError(f"unknown estimator {cfg.estimator!r}, expected gazelle, gaze_anywhere or cached")
