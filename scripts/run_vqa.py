#!/usr/bin/env python3
"""
Answer every G3VQA question with a VLM served by vLLM. The model sees the frame
(or the video, for the temporal questions) with blue gaze arrows drawn from the
tracking predictions. The answers are scored afterwards with scripts/score_vqa.py
and scripts/judge_answers.py.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.config import VQAConfig
import hashlib

import pandas as pd

from source.data.vqa import TEMPORAL_CATEGORIES, load_questions_csv, load_vqa, select_vlm_rows
from source.vqa.backend import HFLocalClient, VLLMClient
from source.vqa.predictions import GazePredictions
from source.vqa.runner import MediaStore, run_vqa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to configs/vqa.yaml")
    ap.add_argument("--vqa-dir", default=None, help="overrides cfg.vqa_dir")
    ap.add_argument("--predictions-csv", default=None, help="overrides cfg.predictions_csv")
    ap.add_argument("--frames-dir", default=None, help="overrides cfg.frames_dir")
    ap.add_argument("--output-csv", default=None, help="overrides cfg.output_csv")
    ap.add_argument("--answer-url", default=None, help="overrides cfg.answer_url")
    ap.add_argument("--answer-model", default=None, help="overrides cfg.answer_model (vLLM served model name)")
    ap.add_argument("--question-format", choices=["json", "csv"], default=None, help="overrides cfg.question_format")
    ap.add_argument("--extended", action="store_true", help="also ask group-engagement and the temporal (whole-video) questions")
    ap.add_argument("--no-temporal", action="store_true", help="drop the whole-video questions (e.g. when the server cannot take video)")
    ap.add_argument("--media-dir", default=None, help="pre-rendered inputs from scripts/render_vqa_inputs.py, overrides cfg.media_dir")
    ap.add_argument("--only-keys", default=None, help="CSV with columns clip, vqa_id: answer only these questions (smoke test)")
    ap.add_argument("--backend", choices=["vllm", "hf-local"], default="vllm", help="hf-local runs --hf-model in-process, one request at a time")
    ap.add_argument("--hf-model", default=None, help="model directory for --backend hf-local")
    ap.add_argument("--clips", default=None, help="comma-separated clip names, default all")
    ap.add_argument("--resume", action="store_true", help="skip questions already in the output CSV")
    args = ap.parse_args()

    cfg = VQAConfig.from_yaml(args.config)
    for key in ("vqa_dir", "predictions_csv", "frames_dir", "output_csv", "answer_url", "answer_model",
                "question_format", "media_dir"):
        if getattr(args, key):
            setattr(cfg, key, getattr(args, key))
    cfg.extended = cfg.extended or args.extended
    if not cfg.answer_model:
        ap.error("set answer_model in the config or pass --answer-model")
    os.environ.setdefault("HF_HOME", os.path.join(cfg.model_cache_dir, "huggingface"))

    clips = args.clips.split(",") if args.clips else None
    loader = load_questions_csv if cfg.question_format == "csv" else load_vqa
    questions = select_vlm_rows(loader(cfg.vqa_dir, clips), extended=cfg.extended)
    if args.no_temporal:
        questions = questions[~questions["question_type"].isin(TEMPORAL_CATEGORIES)].reset_index(drop=True)
    if args.only_keys:
        keys = pd.read_csv(args.only_keys, dtype=str)[["clip", "vqa_id"]]
        questions = questions.assign(vqa_id=questions["vqa_id"].astype(str)).merge(keys, on=["clip", "vqa_id"]).reset_index(drop=True)
    print(f"[vqa] {len(questions)} questions over {questions['clip'].nunique()} clips")
    predictions = GazePredictions(cfg.predictions_csv, cfg.frames_dir, clips)

    if args.backend == "hf-local":
        if not args.hf_model:
            ap.error("--backend hf-local needs --hf-model")
        answerer = HFLocalClient(args.hf_model, cfg.answer_model)
        cfg.max_concurrency = 1
    else:
        answerer = VLLMClient(cfg.answer_url, cfg.answer_model, timeout=cfg.request_timeout, video_transport=cfg.video_transport)
    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest() if path and os.path.isfile(path) else None

    manifest = {
        "args": vars(args), "config": cfg.__dict__, "backend": args.backend, "hf_model": args.hf_model,
        "predictions_csv_sha256": sha(cfg.predictions_csv),
        "code_sha256": {f: sha(os.path.join(os.path.dirname(__file__), "..", f)) for f in (
            "source/vqa/runner.py", "source/vqa/backend.py", "source/vqa/visual_prompt.py", "source/vqa/media.py",
            "source/vqa/video.py", "source/data/vqa.py", "scripts/run_vqa.py")},
        "question_types": questions["question_type"].value_counts().to_dict(),
    }
    asyncio.run(run_vqa(
        questions, predictions, answerer, cfg.output_csv,
        resume=args.resume, max_concurrency=cfg.max_concurrency,
        max_new_tokens=cfg.max_new_tokens,
        media=MediaStore(cfg.media_dir) if cfg.media_dir else None, manifest=manifest,
    ))
    print(f"[vqa] results written to {cfg.output_csv}")


if __name__ == "__main__":
    main()
