#!/usr/bin/env python3
"""
Accuracy of the VLM answers, one number per question group.

  python scripts/score_vqa.py --answers-dir outputs/vqa --judge-dir outputs/judge [--cut 0.5] [--out outputs/scores.csv]

--answers-dir has one answers CSV per model (scripts/run_vqa.py), --judge-dir the matching AnyJev scores (scripts/judge_answers.py).

How an answer is scored
  exact match  the reference is a yes or no, a count, or a frame, and the answer states a verdict (source/vqa/closed_form.py finds it):
               the answer is correct when the verdict equals the reference. A frame answer is correct when its range overlaps a reference range.
  judge        every other answer, including answers that state no verdict: correct when the AnyJev probability is at least --cut.
Questions that were skipped because the person has no gaze point are not counted.

The groups
  Individual   what does one named person look at (gaze object description)
  Ambiguous    a question without an object or person to name: the reference says the description is not unique, or the reference is the
               count 0 ("There is no X in the image", "No one is looking at X")
  Group        counting, shared gaze, exchanged gaze and group engagement, which are about several people at once
  When         temporal questions that ask for a frame
  Count        temporal questions that ask how many times
  Memory       the other temporal questions (which object or person, lists, first and last target, revisiting)
  Overall      all questions
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from source.vqa import closed_form as cf  # noqa: E402

KEYS = ["clip", "person_id", "frame", "vqa_id"]
FRAME_COLS = ["Individual", "Ambiguous", "Group"]
TEMPORAL_COLS = ["When", "Count", "Memory"]
GROUP_TYPES = {"counting", "shared_gaze", "exchanged_gaze", "group_engagement"}
WHEN = {"first_frame_person_object", "nth_look_at_object", "person_appearance", "first_last_look_at_object",
        "first_last_shared_gaze", "look_at_unlooked_object"}
COUNT = {"person_object_count", "person_object_look_count", "face_occlusion_counting", "gaze_exchange_count",
         "majority_object_count", "object_look_count"}
MEMORY = {"gaze_target_revisiting", "person_first_last_target", "person_object_list", "person_object_list_after",
          "person_object_list_before", "face_occlusion_detection", "first_last_looker", "most_looked_object",
          "occluded_gaze_reasoning"}
NOT_UNIQUE = re.compile(r"\bnot unique\b", re.I)
NO_ONE = re.compile(r"^\s*(?:no one|nobody|none)\b", re.I)


def is_ambiguous(reference_answers) -> bool:
    if NOT_UNIQUE.search(reference_answers):
        return True
    route = cf.route(reference_answers)
    if route == "number" and cf.gt_value(route, reference_answers) == 0:
        return True
    return bool(NO_ONE.match(reference_answers.split(cf.SEP)[0]))


def group_of(question_type, template, reference_answers) -> str:
    if question_type in ("temporal_individual", "temporal_social"):
        if template in WHEN:
            return "When"
        if template in COUNT:
            return "Count"
        if template in MEMORY:
            return "Memory"
        raise ValueError(f"unknown temporal template {template!r}")
    if is_ambiguous(reference_answers) or question_type == "absent_object":
        return "Ambiguous"
    if question_type in ("gaze_object_description", "ambiguous_description"):  # an ambiguous-typed row with a real answer is an ordinary person question
        return "Individual"
    if question_type in GROUP_TYPES:
        return "Group"
    raise ValueError(f"unknown question type {question_type!r}")


def score_model(answers_csv, judge_csv, cut):
    cols = KEYS + ["question_type", "template", "question", "reference_answers", "vlm_answer", "skipped_reason"]
    ans = pd.read_csv(answers_csv, dtype=str, keep_default_na=False, usecols=cols)
    ans = ans[ans["skipped_reason"] == ""].drop(columns="skipped_reason").drop_duplicates(subset=KEYS, keep="last")
    jud = pd.read_csv(judge_csv, dtype=str, keep_default_na=False)
    jud = jud[KEYS + ["p_correct"]].drop_duplicates(subset=KEYS, keep="last")
    d = ans.merge(jud, on=KEYS, how="left", validate="one_to_one")
    missing = int((d["p_correct"].isna() | (d["p_correct"] == "")).sum())
    if missing:
        sys.exit(f"{answers_csv}: {missing} answers have no judge score, run scripts/judge_answers.py first")

    exact = []
    for question, refs, answer in zip(d["question"], d["reference_answers"], d["vlm_answer"]):
        route = cf.route(refs)
        if route == "open":
            exact.append(np.nan)
            continue
        ok = cf.check(route, cf.gt_value(route, refs), answer, question)["ok"]
        exact.append(np.nan if ok is None else ok)
    d["exact"] = exact
    p = d["p_correct"].astype(float)
    d["correct"] = np.where(d["exact"].notna(), d["exact"].fillna(0), (p >= cut).astype(int)).astype(int)
    d["group"] = [group_of(q, t, r) for q, t, r in zip(d["question_type"], d["template"], d["reference_answers"])]
    return d


def summarise(d):
    row = {c: 100 * d.loc[d["group"] == c, "correct"].mean() for c in FRAME_COLS + TEMPORAL_COLS}
    temporal = d["group"].isin(TEMPORAL_COLS)
    row["Frame level"] = 100 * d.loc[~temporal, "correct"].mean()
    row["Temporal"] = 100 * d.loc[temporal, "correct"].mean()
    row["Overall"] = 100 * d["correct"].mean()
    row["Exact match share"] = 100 * d["exact"].notna().mean()
    counts = d["group"].value_counts().reindex(FRAME_COLS + TEMPORAL_COLS).fillna(0).astype(int).to_dict()
    counts.update({"Frame level": int((~temporal).sum()), "Temporal": int(temporal.sum()), "Overall": len(d)})
    return row, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers-dir", required=True)
    ap.add_argument("--judge-dir", required=True)
    ap.add_argument("--models", default="", help="comma-separated model names, default all CSVs in --answers-dir")
    ap.add_argument("--cut", type=float, default=0.5, help="an answer judged by AnyJev is correct when P(correct) is at least this")
    ap.add_argument("--out", default="", help="write the table to this CSV")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m] or sorted(f[:-4] for f in os.listdir(args.answers_dir) if f.endswith(".csv"))
    rows, counts = {}, {}
    for model in models:
        d = score_model(os.path.join(args.answers_dir, f"{model}.csv"), os.path.join(args.judge_dir, f"{model}.csv"), args.cut)
        rows[model], counts[model] = summarise(d)
    table = pd.DataFrame(rows).T.sort_values("Overall")
    first = next(iter(counts.values()))
    if any(c != first for c in counts.values()):
        print("warning: the models were not asked the same number of questions")
    pd.set_option("display.width", 200)
    print("questions per group: " + ", ".join(f"{k} {v:,}" for k, v in first.items()))
    print(table.round(1).to_string())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        table.round(2).to_csv(args.out, index_label="model")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
