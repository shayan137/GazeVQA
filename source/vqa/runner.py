"""Answer every question, with many requests in flight.

Results are appended to one CSV (question, references, answer, and what was sent: prompt text,
media file and its hash, arrows, token counts) every `flush_every` rows, so an interrupted run can
be resumed. Every request is also written to a JSONL log next to the CSV, and a run_manifest.json
records the inputs. The answers are scored by scripts/judge_answers.py and scripts/score_vqa.py.

Per-frame questions use a pre-rendered JPEG when `media` has one (scripts/render_vqa_inputs.py)
and are drawn on the fly otherwise; whole-video questions need their pre-rendered mp4.
"""

import asyncio
import json
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from source.vqa.media import media_kind, media_relpath
from source.vqa.visual_prompt import build_overlay, build_prompt

OUTPUT_COLUMNS = [
    "clip", "person_id", "frame", "vqa_id", "question_type", "template", "question", "reference_answers",
    "n_arrows", "arrow_persons", "image_path", "has_overlay",
    "media_kind", "media_path", "media_sha256", "n_video_frames", "video_stride",
    "prompt_text", "answer_model", "vlm_answer", "vlm_latency_s", "vlm_error", "prompt_tokens", "completion_tokens",
    "skipped_reason", "run_id",
]


class MediaStore:
    """Pre-rendered inputs of a render_vqa_inputs.py output directory (may be None)."""

    def __init__(self, root=None):
        self.root = Path(root) if root else None
        self.info = {}  # media_path -> manifest row
        if self.root:
            for name in ("manifest_images.csv", "manifest_videos.csv"):
                f = self.root / name
                if f.exists():
                    for r in pd.read_csv(f, dtype=str, keep_default_na=False).to_dict("records"):
                        if r.get("media_path"):
                            self.info[r["media_path"]] = r

    def path(self, rel: str):
        if self.root and (self.root / rel).is_file():
            return self.root / rel
        return None


def _log_request(log, entry: dict) -> None:
    if log is not None:
        log.write(json.dumps(entry, default=str) + "\n")
        log.flush()


async def answer_row(row, predictions, answerer, max_new_tokens: int, media: MediaStore | None = None,
                     run_id: str = "", log=None) -> dict:
    qtype, pid, frame = row["question_type"], row["person_id"], int(row["frame"])
    kind = media_kind(qtype, pid)
    temporal = kind == "video"
    result = {
        **{k: row[k] for k in ("clip", "person_id", "frame", "vqa_id", "question_type", "question", "reference_answers")},
        "template": row.get("template", ""), "media_kind": "video" if temporal else "image", "run_id": run_id,
        "n_arrows": 0, "arrow_persons": "", "image_path": "", "has_overlay": True,
        "media_path": "", "media_sha256": "", "n_video_frames": None, "video_stride": None, "prompt_text": "",
        "answer_model": answerer.model,
        "vlm_answer": "", "vlm_latency_s": 0.0, "vlm_error": "", "prompt_tokens": None, "completion_tokens": None,
        "skipped_reason": "",
    }
    rel = media_relpath(row["clip"], qtype, pid, frame)
    media_file = media.path(rel) if media else None
    gen_kwargs = {}

    if temporal:
        tracks = predictions.clip_tracks(row["clip"])
        persons = sorted(tracks) if qtype == "temporal_social" else ([pid] if pid in tracks else [])
        result.update(n_arrows=len(persons), arrow_persons=";".join(persons), has_overlay=True)
        if not persons:
            result["skipped_reason"] = "no_gaze_point"
            return result
        if media_file is None:
            result["skipped_reason"] = "missing_media"
            return result
        info = media.info.get(rel, {})
        result.update(media_path=rel, media_sha256=_file_sha256(media_file), n_video_frames=info.get("n_frames_written") or None,
                      video_stride=info.get("stride") or None)
        prompt = build_prompt(row["question"], qtype == "temporal_social", True, temporal=True)
        gen_kwargs = {"video_path": media_file}
    else:
        image_path, arrows, has_overlay, is_scene = predictions.lookup(row["clip"], pid, frame)
        result.update(n_arrows=len(arrows), image_path=image_path or "", has_overlay=has_overlay)
        if image_path is None:
            result["skipped_reason"] = "no_gaze_point"
            return result
        prompt = build_prompt(row["question"], is_scene, has_overlay)
        if media_file is not None:
            result.update(media_path=rel, media_sha256=_file_sha256(media_file))
            gen_kwargs = {"image_path": media_file}
        elif has_overlay:
            gen_kwargs = {"image": await asyncio.to_thread(build_overlay, image_path, arrows)}
        else:
            gen_kwargs = {"image": await asyncio.to_thread(lambda: Image.open(image_path).convert("RGB"))}
    result["prompt_text"] = prompt

    t0 = time.monotonic()
    gen = None
    try:
        gen = await answerer.generate_ex(prompt, max_new_tokens=max_new_tokens, **gen_kwargs)
        result["vlm_answer"] = gen.text
        result["prompt_tokens"], result["completion_tokens"] = gen.prompt_tokens, gen.completion_tokens
    except Exception as e:
        result["vlm_error"] = repr(e)
    result["vlm_latency_s"] = time.monotonic() - t0
    _log_request(log, {
        "run_id": run_id, "clip": row["clip"], "vqa_id": row["vqa_id"], "person_id": pid, "frame": frame,
        "question_type": qtype, "prompt_text": prompt, "media_path": result["media_path"],
        "answer": result["vlm_answer"], "error": result["vlm_error"], "latency_s": result["vlm_latency_s"],
        "prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
        "extra": gen.extra if gen else {},
    })

    return result


async def _run_bounded(rows: list[dict], worker, output_csv: Path, append: bool, max_concurrency: int, flush_every: int = 50):
    sem = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()
    buffer = []
    state = {"first": not append, "done": 0}

    def flush():
        first = state["first"]
        pd.DataFrame(buffer, columns=OUTPUT_COLUMNS).to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
        state["first"] = False
        buffer.clear()

    async def bounded(row):
        async with sem:
            result = await worker(row)
        async with lock:
            buffer.append(result)
            state["done"] += 1
            if state["done"] % 500 == 0:
                print(f"[{state['done']}/{len(rows)}]", flush=True)
            if len(buffer) >= flush_every:
                flush()

    await asyncio.gather(*(bounded(r) for r in rows))
    if buffer:
        flush()


def _done_keys(output_csv: Path) -> set:
    """Questions with a finished row. Rows that failed with a request error are not done, so a
    resumed run asks them again (the new row is appended; readers keep the last row per key)."""
    done = pd.read_csv(output_csv, dtype={"person_id": str, "vqa_id": str}, keep_default_na=False)
    if "vlm_error" in done:
        done = done[done["vlm_error"] == ""]
    return {(r.clip, r.person_id, int(r.frame), str(r.vqa_id)) for r in done.itertuples()}


def _file_sha256(path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(path: Path, info: dict) -> None:
    path.write_text(json.dumps(info, indent=2, default=str))


async def run_vqa(questions: pd.DataFrame, predictions, answerer, output_csv, resume: bool = False,
                  max_concurrency: int = 128, max_new_tokens: int = 4096,
                  media: MediaStore | None = None, manifest: dict | None = None) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    append = resume and output_csv.exists()
    done = _done_keys(output_csv) if append else set()
    rows = [r for r in questions.to_dict("records")
            if (r["clip"], r["person_id"], int(r["frame"]), str(r["vqa_id"])) not in done]
    print(f"{len(rows)} questions to answer ({len(done)} already done)", flush=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    manifest_path = output_csv.with_suffix(".run_manifest.json")
    info = {"run_id": run_id, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "output_csv": str(output_csv),
            "n_questions": len(rows), "n_resumed": len(done), "answer_model": answerer.model, "max_new_tokens": max_new_tokens,
            "max_concurrency": max_concurrency, **(manifest or {})}
    write_run_manifest(manifest_path, info)
    log = open(output_csv.with_suffix(".requests.jsonl"), "a")

    async def worker(row):
        return await answer_row(row, predictions, answerer, max_new_tokens, media, run_id, log)

    await _run_bounded(rows, worker, output_csv, append, max_concurrency)
    log.close()
    await answerer.close()
    out = pd.read_csv(output_csv, keep_default_na=False)
    info.update(finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), rows_in_csv=int(len(out)),
                skipped=out["skipped_reason"].value_counts().to_dict() if "skipped_reason" in out else {},
                errors=int((out["vlm_error"] != "").sum()) if "vlm_error" in out else None)
    write_run_manifest(manifest_path, info)
