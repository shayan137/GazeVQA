#!/usr/bin/env python3
"""Stage 1: run one detector on every test frame (one call per image, as in GTA) and log every box >= 0.1.
Resumable (frames already in detections.jsonl are skipped). Output: outputs/grounding/<detector>/detections.jsonl + run.log"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd, torch
from PIL import Image

from benchmarks.grounding.detectors import DETECTORS

ap = argparse.ArgumentParser()
ap.add_argument("--detector", required=True, choices=list(DETECTORS))
ap.add_argument("--tracks", default="outputs/tracking/page_tracking.csv")
ap.add_argument("--out", default="outputs/grounding")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

out_dir = os.path.join(a.out, a.detector); os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "detections.jsonl")
frames = pd.read_csv(a.tracks, dtype={"person_id": str}, usecols=["clip", "frame", "image_path"]).drop_duplicates(["clip", "frame"]).sort_values(["clip", "frame"])
if a.limit: frames = frames.head(a.limit)
done = set()
if os.path.exists(path):
    with open(path) as f:
        for line in f:
            try: r = json.loads(line); done.add((r["clip"], r["frame"]))
            except Exception: pass  # a half-written last line is re-done
todo = [r for r in frames.itertuples(index=False) if (r.clip, int(r.frame)) not in done]
log = open(os.path.join(out_dir, "run.log"), "a")
def say(m):
    m = f"[{time.strftime('%F %T')}] {m}"; print(m, flush=True); log.write(m + "\n"); log.flush()
say(f"detector={a.detector} frames={len(frames)} done={len(done)} todo={len(todo)} torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
det = DETECTORS[a.detector]()
t0 = time.time()
with open(path, "a") as f:
    for i, r in enumerate(todo, 1):
        t = time.perf_counter()
        img = Image.open(r.image_path).convert("RGB")
        d = det.detect(img)
        ms = (time.perf_counter() - t) * 1000
        f.write(json.dumps({"clip": r.clip, "frame": int(r.frame), "w": img.size[0], "h": img.size[1], "ms": round(ms, 1),
                            "boxes": [[round(float(v), 1) for v in b] for b in d.boxes], "scores": [round(float(s), 4) for s in d.scores],
                            "classes": d.classes}) + "\n")
        if i % 250 == 0 or i == len(todo):
            f.flush(); el = time.time() - t0
            say(f"{i}/{len(todo)} frames, {el/i*1000:.0f} ms/frame, eta {(len(todo)-i)*el/i/60:.1f} min")
say(f"finished {a.detector}: {len(todo)} frames in {(time.time()-t0)/60:.1f} min")
