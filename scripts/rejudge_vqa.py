#!/usr/bin/env python3
"""
judge an existing answer CSV again (e.g. one written with --no-judge)
without re-running the answering model
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source.config import VQAConfig
from source.vqa.backend import VLLMClient
from source.vqa.runner import rejudge


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to configs/vqa.yaml")
    ap.add_argument("--input-csv", required=True, help="answers from scripts/run_vqa.py")
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--judge-url", default=None, help="overrides cfg.judge_url")
    ap.add_argument("--judge-model", default=None, help="overrides cfg.judge_model")
    args = ap.parse_args()

    cfg = VQAConfig.from_yaml(args.config)
    cfg.judge_url = args.judge_url or cfg.judge_url
    cfg.judge_model = args.judge_model or cfg.judge_model
    os.environ.setdefault("HF_HOME", os.path.join(cfg.model_cache_dir, "huggingface"))

    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer(cfg.embedding_model, device="cpu")
    judge = VLLMClient(cfg.judge_url, cfg.judge_model, timeout=cfg.request_timeout)
    asyncio.run(rejudge(
        args.input_csv, judge, embed_model, args.output_csv,
        max_concurrency=cfg.max_concurrency, judge_max_new_tokens=cfg.judge_max_new_tokens,
    ))
    print(f"[rejudge] results written to {args.output_csv}")


if __name__ == "__main__":
    main()
