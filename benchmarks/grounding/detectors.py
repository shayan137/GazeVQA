"""Object-candidate detectors for gaze-target grounding, as used by GTA
(RF-DETR 2XL, YOLO26) plus WeDetect-Uni. Every detector exposes
`detect(PIL.Image) -> Detections` with boxes in pixel xyxy, a score and a
class name, so the GTA selection rule can be applied to any of them.

Checkpoints are read from GROUNDING_CKPTS (default model_cache/grounding). RF-DETR 2XL is
downloaded by the rfdetr library into ~/.roboflow/models/rf-detr-xxlarge.pth.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

CKPT_DIR = os.environ.get("GROUNDING_CKPTS", os.path.join("model_cache", "grounding"))
WEDETECT_REPO = os.environ.get("WEDETECT_REPO", "WeDetect")

# Detections below this are kept in the cache; the selection step filters at the GTA threshold (0.3).
CACHE_THRESHOLD = 0.1


@dataclass
class Detections:
    boxes: np.ndarray    # (N, 4) float32, pixel xyxy
    scores: np.ndarray   # (N,) float32
    classes: list        # N class names ("object" for class-agnostic proposals)


class RFDETR2XL:
    name = "rfdetr_2xl"

    def __init__(self, threshold: float = CACHE_THRESHOLD):
        from rfdetr import RFDETR2XLarge

        self.threshold = threshold
        self.model = RFDETR2XLarge(num_classes=90)

    def detect(self, image: Image.Image) -> Detections:
        det = self.model.predict(image, threshold=self.threshold)
        names = [str(n) for n in det.data["class_name"]] if len(det.xyxy) else []
        return Detections(np.asarray(det.xyxy, dtype=np.float32).reshape(-1, 4), np.asarray(det.confidence, dtype=np.float32), names)


class YOLO26x:
    name = "yolo26x"

    def __init__(self, threshold: float = CACHE_THRESHOLD, device: str = "cuda"):
        from ultralytics import YOLO

        self.model = YOLO(os.path.join(CKPT_DIR, "yolo26x.pt"))
        self.threshold, self.device = threshold, device

    def detect(self, image: Image.Image) -> Detections:
        r = self.model.predict(image, device=self.device, conf=self.threshold, verbose=False)[0]
        names = [r.names[int(c)] for c in r.boxes.cls.tolist()]
        return Detections(r.boxes.xyxy.cpu().numpy().astype(np.float32).reshape(-1, 4), r.boxes.conf.cpu().numpy().astype(np.float32), names)


class WeDetectLargeUni:
    """WeDetect-Large-Uni: class-agnostic object proposals (no vocabulary), 1280 px letterbox input."""
    name = "wedetect_large_uni"

    def __init__(self, threshold: float = CACHE_THRESHOLD, num_proposals: int = 300):
        sys.path.insert(0, WEDETECT_REPO)
        import generate_proposal as gp  # self-contained (torch only), no mmdet needed

        self.threshold = threshold
        model = gp.SimpleYOLOWorldDetector(backbone_size="large", prompt_dim=768, num_prompts=256, num_proposals=num_proposals)
        ckpt = torch.load(os.path.join(CKPT_DIR, "wedetect", "wedetect_large_uni.pth"), map_location="cpu")
        for key in list(ckpt):
            if "backbone" in key:
                ckpt[key.replace("backbone.image_model.model.", "backbone.")] = ckpt.pop(key)
        for key in list(ckpt):
            if "bbox_head" in key:
                new = key.replace("bbox_head.head_module.", "bbox_head.")
                for a, b in (("0.2.", "0.6."), ("1.2.", "1.6."), ("2.2.", "2.6."), ("1.bn", "4"), ("1.conv", "3"), ("0.bn", "1"), ("0.conv", "0")):
                    new = new.replace(a, b)
                ckpt[new] = ckpt.pop(key)
        msg = model.load_state_dict(ckpt, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print(f"[wedetect] load_state_dict: missing={msg.missing_keys[:5]} unexpected={msg.unexpected_keys[:5]}", flush=True)
        self.model = model.cuda().eval()

    def detect(self, image: Image.Image) -> Detections:
        with torch.no_grad():
            out = self.model([image])[0]
        boxes, scores = out["bboxes"].float().cpu(), out["scores"].float().cpu()
        keep = scores > self.threshold
        boxes, scores = boxes[keep].numpy().astype(np.float32), scores[keep].numpy().astype(np.float32)
        return Detections(boxes.reshape(-1, 4), scores, ["object"] * len(scores))


DETECTORS = {"rfdetr_2xl": RFDETR2XL, "yolo26x": YOLO26x, "wedetect_large_uni": WeDetectLargeUni}
