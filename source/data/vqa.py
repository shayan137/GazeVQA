"""G3VQA question files: one <clip>.json per clip with a "vqa" list of
{id, frame, person_id, question, answers}.

person_id is a person index ("1", "2", ...) for person-specific questions,
"all" for scene-level questions, and "unique" for ambiguous-description
questions, where the description matches more than one person.
"""

import json
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
# Answered directly from the predicted gaze point, no VLM call.
DIRECT_ANSWER_CATEGORIES = {"gaze_point_coordinates", "gaze_direction"}

AMBIGUOUS_IDS = ("unique", "uniqe")
SCENE_IDS = ("all", "")


def classify_question(question: str, person_id: str, answers: list[str]) -> str:
    ql = question.lower()
    if person_id in AMBIGUOUS_IDS:
        return "ambiguous_description"
    if person_id in SCENE_IDS:
        if ql.startswith("how many people") and any(a.strip().startswith("0") for a in answers):
            return "absent_object"
        if ql.startswith("how many people") and "same object" in ql:
            return "shared_gaze"
        if ql.startswith("how many people"):
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
        person_id = str(e.get("person_id") or "")
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


def select_vlm_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["question_type"].isin(VLM_CATEGORIES)].reset_index(drop=True)
