"""OpenFace 3.0 head/eye direction (yaw, pitch) for every labelled head crop.

Only the multitask model (model/MLT.py) of the OpenFace 3.0 repo is used;
head boxes come from the labels, so no face detector is needed.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_openface(openface_root: str, checkpoint: str, device: str = "cuda") -> torch.nn.Module:
    root = str(Path(openface_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    model = importlib.import_module("model.MLT").MLT()
    state = torch.load(Path(checkpoint).expanduser(), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _crop_box(bbox, image_size) -> tuple[int, int, int, int]:
    x, y, w, h = (float(v) for v in bbox)
    if not (all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0):
        raise ValueError("invalid head box")
    W, H = image_size
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(W, math.ceil(x + w)), min(H, math.ceil(y + h))
    if right <= left or bottom <= top:
        raise ValueError("empty head box")
    return left, top, right, bottom


class HeadCrops(Dataset):
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        try:
            with Image.open(row["image_path"]) as im:
                image = ImageOps.exif_transpose(im).convert("RGB")
            bbox = (row["bbox_x"], row["bbox_y"], row["bbox_width"], row["bbox_height"])
            return TRANSFORM(image.crop(_crop_box(bbox, image.size))), i, True
        except Exception:
            # missing frame or invalid box, kept as an invalid row
            return torch.zeros(3, 224, 224), i, False


def _collate(batch):
    images, idx, ok = zip(*batch)
    return torch.stack(images), list(idx), list(ok)


def head_pose(rows: pd.DataFrame, model: torch.nn.Module, device: str = "cuda", batch_size: int = 128, workers: int = 2) -> pd.DataFrame:
    """Returns rows with added yaw, pitch (radians) and valid columns."""
    loader = DataLoader(
        HeadCrops(rows), batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device != "cpu", persistent_workers=workers > 0, collate_fn=_collate,
    )
    yaw = np.full(len(rows), np.nan, dtype=np.float32)
    pitch = np.full(len(rows), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for images, idx, ok in tqdm(loader, desc="openface"):
            keep = [j for j, v in enumerate(ok) if v]
            if not keep:
                continue
            gaze = model(images[keep].to(device, non_blocking=True))[1].float().cpu().numpy()
            for out, j in zip(gaze, keep):
                yaw[idx[j]], pitch[idx[j]] = out

    out = rows.reset_index(drop=True).copy()
    out["yaw"], out["pitch"] = yaw, pitch
    out["valid"] = np.isfinite(yaw) & np.isfinite(pitch)
    return out
