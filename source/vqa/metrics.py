"""Answer metrics. Every metric is computed against each reference answer and
the best match is kept.

- judge: 0-5 score from the LLM judge (source/vqa/judge.py)
- embedding similarity: cosine similarity of sentence embeddings
- BLEU-4 (smoothed), ROUGE-L F1 and token F1 on lower-cased word tokens
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from source.data.vqa import REFERENCE_ANSWER_SEP

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def split_references(reference_answers) -> list[str]:
    return [r for r in str(reference_answers).split(REFERENCE_ANSWER_SEP) if r]


def embedding_similarity(candidate: str, reference_answers: str, model) -> float:
    refs = split_references(reference_answers)
    if not refs or pd.isna(candidate):
        return float("nan")
    emb = model.encode([candidate] + refs, normalize_embeddings=True)
    return float(np.max(emb[1:] @ emb[0]))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def bleu4(cand: list[str], ref: list[str]) -> float:
    if not cand:
        return 0.0
    c_len, r_len = len(cand), len(ref)
    log_p = []
    for n in range(1, 5):
        cand_ngrams = Counter(tuple(cand[i:i + n]) for i in range(max(0, c_len - n + 1)))
        ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(max(0, r_len - n + 1)))
        overlap = sum(min(c, ref_ngrams[g]) for g, c in cand_ngrams.items())
        total = max(c_len - n + 1, 0)
        if total == 0:
            return 0.0
        # smoothing for n-gram orders with no overlap, common for short answers
        p = overlap / total if overlap else 1.0 / (2 ** n) / total
        log_p.append(math.log(p))
    bp = 1.0 if c_len > r_len else math.exp(1 - r_len / c_len)
    return bp * math.exp(sum(log_p) / 4)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _f1(overlap: int, n_cand: int, n_ref: int) -> float:
    if overlap == 0:
        return 0.0
    p, r = overlap / n_cand, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_l(cand: list[str], ref: list[str]) -> float:
    if not cand or not ref:
        return 0.0
    return _f1(_lcs(cand, ref), len(cand), len(ref))


def token_f1(cand: list[str], ref: list[str]) -> float:
    if not cand and not ref:
        return 1.0
    if not cand or not ref:
        return 0.0
    return _f1(sum((Counter(cand) & Counter(ref)).values()), len(cand), len(ref))


def text_scores(candidate: str, reference_answers: str) -> tuple[float, float, float]:
    """(BLEU-4, ROUGE-L, token F1). Empty answers score 0."""
    refs = split_references(reference_answers)
    if not refs or pd.isna(candidate) or not str(candidate).strip():
        return 0.0, 0.0, 0.0
    cand = tokenize(candidate)
    best = [0.0, 0.0, 0.0]
    for ref in refs:
        ref = tokenize(ref)
        best = [max(best[0], bleu4(cand, ref)), max(best[1], rouge_l(cand, ref)), max(best[2], token_f1(cand, ref))]
    return tuple(best)


def add_text_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = [text_scores(c, r) for c, r in zip(df["vlm_answer"], df["reference_answers"])]
    df["bleu4"], df["rouge_l"], df["token_f1"] = zip(*scores) if scores else ([], [], [])
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Judge mean is over judged rows. The >=4 rate is over all rows, so a
    question that got no answer counts as wrong."""
    judged = df["judge_score"].dropna()
    out = {
        "rows": int(len(df)),
        "judged": int(len(judged)),
        "judge": float(judged.mean()) if len(judged) else float("nan"),
        "ge4": float((df["judge_score"] >= 4).mean() * 100),
        "embed": float(df["embedding_similarity"].mean()),
    }
    for col in ("bleu4", "rouge_l", "token_f1"):
        if col in df:
            out[col] = float(df[col].mean())
    return out


def summarize_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({qtype: summarize(g) for qtype, g in df.groupby("question_type")}).T
