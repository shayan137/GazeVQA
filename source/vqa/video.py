"""Whole-clip videos with gaze arrows for the temporal questions.

The answers to temporal questions are clip frame numbers ("Frame 491"), while video
VLMs reason in seconds. Every written frame therefore carries its clip frame number
in the top-left corner.

The container fps is 2, not the real rate. vLLM's Qwen3-VL / Qwen3.5 (and Molmo2) video
processors resample every video to 2 fps by default, so a file at native_fps / stride (e.g.
15 fps) lost 7 of every 8 written frames: measured with Qwen3.5-0.8B, a 64-frame video cost
2,125 prompt tokens (8 frames) instead of about 17,000. At fps 2 the resampling keeps every
written frame. The "<t seconds>" timestamps then count frames at 0.5 s steps and the real
timing is only in the manifest (true_fps); the burned-in frame number is what answers use.
"""

import hashlib
import math
import re
from pathlib import Path

import cv2
import numpy as np

from source.vqa.visual_prompt import ARROW_COLOR_BGR, ARROW_WIDTH_PX, draw_gaze_arrow

_FRAME_RE = re.compile(r"_(\d+)\.jpg$")


def clip_frame_files(frames_dir, clip: str) -> dict[int, Path]:
    """{frame number: jpg path} of one clip, frames are 1-based and clip-local."""
    out = {}
    for p in Path(frames_dir, clip).glob(f"{clip}_*.jpg"):
        m = _FRAME_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


def sample_frame_numbers(frames: list[int], cap: int) -> tuple[list[int], int]:
    """Uniform stride so that at most `cap` frames (plus the last one) are written."""
    n = len(frames)
    stride = 1 if n <= cap else math.ceil((n - 1) / (cap - 1))
    picked = frames[::stride]
    if picked[-1] != frames[-1]:
        picked.append(frames[-1])
    return picked, stride


def draw_frame_counter(frame, number: int) -> None:
    h = frame.shape[0]
    scale = max(0.6, h / 540.0)
    thickness = max(1, round(scale * 2))
    text = f"frame {number}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = int(8 * scale)
    cv2.rectangle(frame, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame, text, (pad, th + pad), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


CONTAINER_FPS = 2.0


def render_clip_video(clip: str, frame_files: dict[int, Path], tracks: dict, persons: list[str], out_path,
                      cap: int = 64, native_fps: float = 30.0, max_width: int = 960,
                      container_fps: float = CONTAINER_FPS) -> dict:
    """Writes the mp4 and returns what was written. `tracks` is
    {person_id: {frame: (head_xy, gaze_xy)}} in source pixels; `persons` picks whose arrows are drawn."""
    frames = list(frame_files)
    picked, stride = sample_frame_numbers(frames, cap)
    first = cv2.imread(str(frame_files[picked[0]]))
    h0, w0 = first.shape[:2]
    scale = min(1.0, max_width / w0)
    w, h = int(round(w0 * scale / 2) * 2), int(round(h0 * scale / 2) * 2)  # even sizes for the codec
    true_fps = native_fps / stride  # real frame rate of the written frames

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), container_fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {out_path}")
    missing_arrow = 0
    for number in picked:
        img = cv2.imread(str(frame_files[number]))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        sx, sy = w / w0, h / h0
        for pid in persons:
            arrow = tracks.get(pid, {}).get(number)
            if arrow is None:
                missing_arrow += 1
                continue
            (hx, hy), (gx, gy) = arrow
            draw_gaze_arrow(img, (hx * sx, hy * sy), (gx * sx, gy * sy), ARROW_COLOR_BGR, ARROW_WIDTH_PX)
        draw_frame_counter(img, number)
        writer.write(img)
    writer.release()

    data = out_path.read_bytes()
    return {
        "clip": clip, "persons": ";".join(persons), "n_frames_clip": len(frames), "n_frames_written": len(picked),
        "stride": stride, "fps": container_fps, "true_fps": round(true_fps, 4), "native_fps_assumed": native_fps, "width": w, "height": h,
        "src_width": w0, "src_height": h0, "frame_first": picked[0], "frame_last": picked[-1],
        "person_frames_without_arrow": missing_arrow, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }
