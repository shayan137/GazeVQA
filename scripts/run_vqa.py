#!/usr/bin/env python3
"""
answer every G3VQA question with a VLM, using the gaze-arrow visual prompt
built from the tracking predictions, then judge each answer with a text-only
LLM and compute embedding similarity. Both models are served with vLLM.
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.config import VQAConfig
from source.data.vqa import load_vqa, select_vlm_rows
from source.vqa.backend import VLLMClient
from source.vqa.predictions import GazePredictions
from source.vqa.runner import run_vqa


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to configs/vqa.yaml")
    ap.add_argument("--vqa-dir", default=None, help="overrides cfg.vqa_dir")
    ap.add_argument("--predictions-csv", default=None, help="overrides cfg.predictions_csv")
    ap.add_argument("--frames-dir", default=None, help="overrides cfg.frames_dir")
    ap.add_argument("--output-csv", default=None, help="overrides cfg.output_csv")
    ap.add_argument("--answer-url", default=None, help="overrides cfg.answer_url")
    ap.add_argument("--answer-model", default=None, help="overrides cfg.answer_model (vLLM served model name)")
    ap.add_argument("--judge-url", default=None, help="overrides cfg.judge_url")
    ap.add_argument("--judge-model", default=None, help="overrides cfg.judge_model")
    ap.add_argument("--no-judge", action="store_true", help="answer only, judge later with scripts/rejudge_vqa.py")
    ap.add_argument("--clips", default=None, help="comma-separated clip names, default all")
    ap.add_argument("--resume", action="store_true", help="skip questions already in the output CSV")
    args = ap.parse_args()

    cfg = VQAConfig.from_yaml(args.config)
    for key in ("vqa_dir", "predictions_csv", "frames_dir", "output_csv", "answer_url", "answer_model", "judge_url", "judge_model"):
        if getattr(args, key):
            setattr(cfg, key, getattr(args, key))
    if not cfg.answer_model:
        ap.error("set answer_model in the config or pass --answer-model")
    os.environ.setdefault("HF_HOME", os.path.join(cfg.model_cache_dir, "huggingface"))

    clips = args.clips.split(",") if args.clips else None
    questions = select_vlm_rows(load_vqa(cfg.vqa_dir, clips))
    print(f"[vqa] {len(questions)} questions over {questions['clip'].nunique()} clips")
    predictions = GazePredictions(cfg.predictions_csv, cfg.frames_dir, clips)

    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(cfg.embedding_model, device="cpu")
    answerer = VLLMClient(cfg.answer_url, cfg.answer_model, timeout=cfg.request_timeout)
    judge = None
    if cfg.judge_model and not args.no_judge:
        judge = VLLMClient(cfg.judge_url, cfg.judge_model, timeout=cfg.request_timeout)

    asyncio.run(run_vqa(
        questions, predictions, answerer, judge, embed_model, cfg.output_csv,
        resume=args.resume, max_concurrency=cfg.max_concurrency,
        max_new_tokens=cfg.max_new_tokens, judge_max_new_tokens=cfg.judge_max_new_tokens,
    ))
    print(f"[vqa] results written to {cfg.output_csv}")


if __name__ == "__main__":
    main()
