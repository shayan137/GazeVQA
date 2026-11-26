#!/usr/bin/env python3
"""
extract ChildPlay clip frames as <frames_dir>/<clip>/<clip>_<frame:06d>.jpg
(clip-relative, 1-indexed). Frame selection follows ChildPlay's official
extract-clips-from-videos.py, including its downsampled and one-frame-lag clips.
"""
import argparse
import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

# Clips the official script extracts with a one-frame lag to match the annotations.
LAGGED_CLIPS = {
    "smwfiZd8HLc_7508-8408-downsampled", "smwfiZd8HLc_10346-12974-downsampled",
    "smwfiZd8HLc_16176-16586-downsampled", "smwfiZd8HLc_20322-20668-downsampled",
    "31lG75MDwSA_2857-2928", "31lG75MDwSA_3180-3306", "31lG75MDwSA_4686-4764",
    "EuYseDl2jm8_1869-2033", "EuYseDl2jm8_3680-3791",
}


def extract_clip(video_path: str, clip: str, frame_count: int, out_dir: Path) -> int:
    interval = clip.rsplit("_", 1)[1]
    downsampled = "downsampled" in interval
    frame_start = int(interval.replace("-downsampled", "").split("-")[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start - (2 if clip in LAGGED_CLIPS else 1))

    n = 0
    ret, frame = cap.read()
    while ret:
        n += 1
        if downsampled:  # 60 fps source, keep every other frame
            ret, frame = cap.read()
        cv2.imwrite(str(out_dir / f"{clip}_{n:06d}.jpg"), frame)
        ret, frame = cap.read()
        if n == frame_count:
            break
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-csv", required=True, help="ChildPlay clips.csv")
    ap.add_argument("--videos-dir", required=True, help="ChildPlay source videos (<video_id>.mp4)")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--labels-dir", default=None, help="only extract clips that have a label CSV here")
    args = ap.parse_args()

    clips = pd.read_csv(args.clips_csv)
    if args.labels_dir:
        wanted = {p.stem for p in Path(args.labels_dir).glob("*.csv")}
        clips = clips[clips["clip"].isin(wanted)]

    for row in tqdm(clips.itertuples(), total=len(clips)):
        video = os.path.join(args.videos_dir, f"{row.video_id}.mp4")
        extract_clip(video, row.clip, int(row.frame_count), Path(args.frames_dir) / row.clip)


if __name__ == "__main__":
    main()
