#!/usr/bin/env python3
"""
render every model input of the gaze-arrow VQA once, so the answer jobs only read files:
one overlaid JPEG per per-frame question (one arrow for a person question, all arrows for a
scene question, the plain frame for an ambiguous description) and one mp4 per whole-video
question (the whole clip with only that person's arrow, or with everyone's arrows).
Writes manifest_images.csv and manifest_videos.csv next to the files.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from source.data.vqa import load_questions_csv, select_vlm_rows
from source.vqa.media import media_kind, media_relpath
from source.vqa.predictions import GazePredictions
from source.vqa.video import clip_frame_files, render_clip_video
from source.vqa.visual_prompt import build_overlay

_PRED = None


def _init(pred_csv, frames_dir, clips):
    global _PRED
    _PRED = GazePredictions(pred_csv, frames_dir, clips)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def render_image(job):
    """job: (kind, clip, frame, person_id, relpath, out_root). Returns a manifest row."""
    kind, clip, frame, person_id, rel, out_root = job
    lookup_person = {"individual": person_id, "scene": "all", "plain": "unique"}[kind]
    image_path, arrows, _, _ = _PRED.lookup(clip, lookup_person, frame)
    row = {"kind": kind, "clip": clip, "frame": frame, "person_id": person_id if kind == "individual" else "",
           "n_arrows": len(arrows), "source_path": image_path or "", "media_path": "", "bytes": 0,
           "sha256": "", "skipped_reason": ""}
    if image_path is None:
        row["skipped_reason"] = "no_gaze_point"
        return row
    out = os.path.join(out_root, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if kind == "plain":
        shutil.copyfile(image_path, out)
    else:
        build_overlay(image_path, arrows).save(out, format="JPEG", quality=90)
    row.update(media_path=rel, bytes=os.path.getsize(out), sha256=_sha(out))
    return row


def render_video(job):
    """job: (clip, who, relpath, frames_dir, out_root, cap, fps, width); who is 'all' or a person id."""
    clip, who, rel, frames_dir, out_root, cap, fps, width, container_fps = job
    tracks = _PRED.clip_tracks(clip)
    persons = sorted(tracks) if who == "all" else [who]
    row = {"kind": "video_all" if who == "all" else "video_person", "clip": clip, "person_id": "" if who == "all" else who,
           "media_path": rel, "skipped_reason": ""}
    files = clip_frame_files(frames_dir, clip)
    if not files:
        row["skipped_reason"] = "no_frames"
        return row
    if who != "all" and who not in tracks:
        row["skipped_reason"] = "no_gaze_point"
        return row
    row.update(render_clip_video(clip, files, tracks, persons, os.path.join(out_root, rel), cap, fps, width, container_fps))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions-dir", required=True, help="annotation package csv/test (<clip>_VQA.csv, <clip>_temporal.csv)")
    ap.add_argument("--predictions-csv", required=True, help="tracking-agent output of the gaze estimator")
    ap.add_argument("--frames-dir", required=True, help="<frames-dir>/<clip>/<clip>_<frame:06d>.jpg")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--images", action="store_true", help="render the per-frame images")
    ap.add_argument("--videos", action="store_true", help="render the temporal videos")
    ap.add_argument("--clips", default=None, help="comma-separated clip names, default all")
    ap.add_argument("--only-keys", default=None, help="CSV with columns clip, vqa_id: render only the inputs these questions need")
    ap.add_argument("--max-frames", type=int, default=64, help="frame cap per video (uniform stride above it)")
    ap.add_argument("--native-fps", type=float, default=30.0, help="frame rate of the clip frames (ChildPlay: 30), recorded as true_fps")
    ap.add_argument("--container-fps", type=float, default=2.0, help="fps written into the mp4; 2 keeps every frame under vLLM's default 2 fps resampling")
    ap.add_argument("--max-width", type=int, default=960, help="video width cap in pixels")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not (args.images or args.videos):
        ap.error("pass --images and/or --videos")

    clips = args.clips.split(",") if args.clips else None
    q = select_vlm_rows(load_questions_csv(args.questions_dir, clips), extended=True)
    if args.only_keys:
        keys = pd.read_csv(args.only_keys, dtype=str)
        q = q.merge(keys[["clip", "vqa_id"]], on=["clip", "vqa_id"])
    q["frame"] = q["frame"].astype(int)
    q["kind"] = [media_kind(t, p) for t, p in zip(q["question_type"], q["person_id"])]
    q["rel"] = [media_relpath(c, t, p, f) for c, t, p, f in zip(q["clip"], q["question_type"], q["person_id"], q["frame"])]
    print(f"[render] {len(q)} questions: {q['kind'].value_counts().to_dict()}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    init = (args.predictions_csv, args.frames_dir, clips)
    t0 = time.time()

    if args.images:
        im = q[q["kind"] != "video"].drop_duplicates("rel")
        jobs = [(k, c, f, p, r, args.out_dir) for k, c, f, p, r in zip(im["kind"], im["clip"], im["frame"], im["person_id"], im["rel"])]
        print(f"[render] {len(jobs)} distinct images", flush=True)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
            rows = list(ex.map(render_image, jobs, chunksize=64))
        man = pd.DataFrame(rows)
        man.to_csv(os.path.join(args.out_dir, "manifest_images.csv"), index=False)
        print(f"[render] images: {man['media_path'].ne('').sum()} written, "
              f"{man['skipped_reason'].ne('').sum()} skipped ({man['skipped_reason'].value_counts().to_dict()}), "
              f"{man['bytes'].sum() / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)

    if args.videos:
        vd = q[q["kind"] == "video"].drop_duplicates("rel")
        jobs = []
        for clip, rel, qt, pid in zip(vd["clip"], vd["rel"], vd["question_type"], vd["person_id"]):
            jobs.append((clip, "all" if qt == "temporal_social" else pid, rel, args.frames_dir, args.out_dir,
                         args.max_frames, args.native_fps, args.max_width, args.container_fps))
        print(f"[render] {len(jobs)} distinct videos", flush=True)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
            rows = list(ex.map(render_video, jobs, chunksize=1))
        man = pd.DataFrame(rows)
        man.to_csv(os.path.join(args.out_dir, "manifest_videos.csv"), index=False)
        print(f"[render] videos: {man['sha256'].notna().sum()} written, skipped {man['skipped_reason'].value_counts().to_dict()}, "
              f"{man.get('bytes', pd.Series(dtype=float)).sum() / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
