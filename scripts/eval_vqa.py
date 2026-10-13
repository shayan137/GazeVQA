#!/usr/bin/env python3
"""
summarize VQA result CSVs: mean judge score, % of answers scored >= 4,
embedding similarity, BLEU-4, ROUGE-L and token F1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from source.vqa.metrics import add_text_scores, summarize, summarize_by_type

COLUMNS = [("judge", "Judge", ".3f"), ("ge4", ">=4 (%)", ".1f"), ("embed", "Embed.", ".3f"),
           ("bleu4", "BLEU-4", ".3f"), ("rouge_l", "ROUGE-L", ".3f"), ("token_f1", "Token-F1", ".3f")]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"person_id": str}).rename(columns={"rouge_l_f1": "rouge_l"})
    if "bleu4" not in df:
        df = add_text_scores(df)
    return df


def print_table(names, summaries):
    width = max(len(n) for n in names) + 2
    print(f"{'':<{width}}" + "".join(f"{title:>10}" for _, title, _ in COLUMNS))
    for name, s in zip(names, summaries):
        print(f"{name:<{width}}" + "".join(f"{s[key]:>10{fmt}}" for key, _, fmt in COLUMNS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result CSVs from scripts/run_vqa.py")
    ap.add_argument("--by-type", action="store_true", help="also print scores per question type")
    args = ap.parse_args()

    names, frames = [], []
    for path in args.results:
        names.append(os.path.basename(path).split("__")[0].removesuffix(".csv"))
        frames.append(load(path))
    print_table(names, [summarize(df) for df in frames])

    if args.by_type:
        for name, df in zip(names, frames):
            by_type = summarize_by_type(df)
            print(f"\n{name}")
            print_table(list(by_type.index), [row for _, row in by_type.iterrows()])


if __name__ == "__main__":
    main()
