"""W3GazeVQA question files. Two layouts are read: one <clip>.json per clip with a "vqa" list of
{id, frame, person_id, question, answers}, or the annotation export with <clip>_VQA.csv
(clip, frame, person_id, question, answer1..K) and <clip>_temporal.csv for the whole-video questions.

person_id is a person index ("1", "2", ...) for person-specific questions,
"all" for scene-level questions, and "unique" for ambiguous-description
questions, where the description matches more than one person.
"""

import json
import re
from pathlib import Path

import pandas as pd

# Multiple reference answers are stored in one CSV cell with this separator.
REFERENCE_ANSWER_SEP = " ||| "

# Answered by the VLM.
VLM_CATEGORIES = {
    "gaze_object_description",
    "counting",
    "shared_gaze",
    "exchanged_gaze",
    "absent_object",
    "ambiguous_description",
}
# The extended set also asks the group-engagement questions and the whole-video temporal questions. A temporal question about one person gets
# a video with only that person's arrow; every other temporal question gets all arrows.
TEMPORAL_CATEGORIES = {"temporal_individual", "temporal_social"}
VLM_CATEGORIES_EXTENDED = VLM_CATEGORIES | {"group_engagement"} | TEMPORAL_CATEGORIES
# Answered directly from the predicted gaze point, no VLM call.
DIRECT_ANSWER_CATEGORIES = {"gaze_point_coordinates", "gaze_direction"}

AMBIGUOUS_IDS = ("unique", "uniqe")
SCENE_IDS = ("all", "")
NOT_UNIQUE = re.compile(r"\bnot unique\b", re.I)
HOW_MANY = ("how many people", "how many individuals")   # one counting question was typed with the second wording


def ambiguous_person_id(person_id: str, answers: list[str]) -> str:
    """Ambiguous-description questions carry person_id "unique", but a few were saved with "all" or "". Their reference answer
    says "The description is not unique", so they are treated as ambiguous questions too."""
    if person_id not in AMBIGUOUS_IDS and any(NOT_UNIQUE.search(a) for a in answers):
        return "unique"
    return person_id


def classify_question(question: str, person_id: str, answers: list[str]) -> str:
    ql = question.lower()
    if person_id in AMBIGUOUS_IDS:
        return "ambiguous_description"
    if person_id in SCENE_IDS:
        if ql.startswith(HOW_MANY) and any(a.strip().startswith("0") for a in answers):
            return "absent_object"
        if ql.startswith(HOW_MANY) and "same object" in ql:
            return "shared_gaze"
        if ql.startswith(HOW_MANY):
            return "counting"
        if question in ("Is there a shared gaze target in the image?", "What is the main object people are looking at?"):
            return "shared_gaze"
        if question == "Is there exchanged gaze in the image?":
            return "exchanged_gaze"
        if question == "Is there any person not engaged with the group?":
            return "group_engagement"
        return "other"
    if "coordinates" in ql:
        return "gaze_point_coordinates"
    if "directing" in ql and "gaze" in ql:
        return "gaze_direction"
    if "looking at" in ql:
        return "gaze_object_description"
    return "other"


def _load_clip(path: Path) -> list[dict]:
    state = json.loads(path.read_text())
    rows = []
    for e in state.get("vqa", []):
        answers = [a for a in e.get("answers", []) if a and str(a).strip()]
        person_id = ambiguous_person_id(str(e.get("person_id") or ""), [str(a) for a in answers])
        question = e.get("question") or ""
        rows.append({
            "clip": path.stem,
            "vqa_id": e.get("id"),
            "frame": int(e["frame"]),
            "person_id": person_id,
            "question": question,
            "reference_answers": REFERENCE_ANSWER_SEP.join(answers),
            "question_type": classify_question(question, person_id, answers),
        })
    return rows


def load_vqa(vqa_dir, clips: list[str] | None = None) -> pd.DataFrame:
    files = sorted(Path(vqa_dir).glob("*.json"))
    if clips is not None:
        files = [f for f in files if f.stem in set(clips)]
    return pd.DataFrame([r for f in files for r in _load_clip(f)])


def _answers_from_row(row: dict) -> list[str]:
    cols = sorted((c for c in row if c.startswith("answer") and c[6:].isdigit()), key=lambda c: int(c[6:]))
    return [str(row[c]).strip() for c in cols if isinstance(row[c], str) and row[c].strip()]


def load_vqa_csv(csv_dir, clips: list[str] | None = None) -> pd.DataFrame:
    """Per-frame questions from <clip>_VQA.csv. The files have no question id, so
    vqa_id is the row number within the file ("r0", "r1", ...)."""
    files = sorted(Path(csv_dir).glob("*_VQA.csv"))
    rows = []
    for f in files:
        clip = f.name[: -len("_VQA.csv")]
        if clips is not None and clip not in set(clips):
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for i, r in enumerate(df.to_dict("records")):
            question = r["question"]
            answers = _answers_from_row(r)
            person_id = ambiguous_person_id((r.get("person_id") or "").strip(), answers)
            rows.append({
                "clip": clip, "vqa_id": f"r{i}", "frame": int(r["frame"]), "person_id": person_id,
                "question": question, "reference_answers": REFERENCE_ANSWER_SEP.join(answers),
                "question_type": classify_question(question, person_id, answers),
            })
    return pd.DataFrame(rows)


def temporal_person(params: str) -> str:
    """The single person a temporal question is about ("person=3; object=..."), else ""."""
    for part in str(params or "").split(";"):
        key, _, value = part.partition("=")
        if key.strip() == "person" and value.strip().isdigit():
            return value.strip()
    return ""


def load_temporal_csv(csv_dir, clips: list[str] | None = None) -> pd.DataFrame:
    """Whole-video questions from <clip>_temporal.csv. frame is 0 (the window is the whole
    clip); person_id is the single person the question is about, or "all"."""
    rows = []
    for f in sorted(Path(csv_dir).glob("*_temporal.csv")):
        clip = f.name[: -len("_temporal.csv")]
        if clips is not None and clip not in set(clips):
            continue
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for r in df.to_dict("records"):
            person = temporal_person(r.get("params"))
            rows.append({
                "clip": clip, "vqa_id": r["qid"], "frame": 0, "person_id": person or "all",
                "question": r["question"], "reference_answers": r["answer"].strip(),
                "question_type": "temporal_individual" if person else "temporal_social",
                "template": r["template"], "params": r["params"], "evidence": r["evidence"],
            })
    return pd.DataFrame(rows)


def load_questions_csv(csv_dir, clips: list[str] | None = None) -> pd.DataFrame:
    """Per-frame and temporal questions of one annotation package (csv/test)."""
    return pd.concat([load_vqa_csv(csv_dir, clips), load_temporal_csv(csv_dir, clips)], ignore_index=True)


def select_vlm_rows(df: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    keep = VLM_CATEGORIES_EXTENDED if extended else VLM_CATEGORIES
    return df[df["question_type"].isin(keep)].reset_index(drop=True)
