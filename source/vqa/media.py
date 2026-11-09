"""Pre-rendered model inputs: which image or video file belongs to which question.

Both the renderer (scripts/render_vqa_inputs.py) and the runner use these paths, so
a question finds its file without a lookup table:

  images/<clip>/<clip>_<frame:06d>_p<person>.jpg   one person's arrow (individual question)
  images/<clip>/<clip>_<frame:06d>_all.jpg         every person's arrow (scene-level question)
  images/<clip>/<clip>_<frame:06d>_plain.jpg       no arrow (ambiguous description)
  videos/<clip>/<clip>_p<person>.mp4               whole clip, only that person's arrow
  videos/<clip>/<clip>_all.mp4                     whole clip, every person's arrow
"""

from source.data.vqa import AMBIGUOUS_IDS, SCENE_IDS, TEMPORAL_CATEGORIES


def media_kind(question_type: str, person_id: str) -> str:
    """'video', or the image kind 'individual' / 'scene' / 'plain'."""
    if question_type in TEMPORAL_CATEGORIES:
        return "video"
    if person_id in AMBIGUOUS_IDS:
        return "plain"
    if person_id in SCENE_IDS:
        return "scene"
    return "individual"


def media_relpath(clip: str, question_type: str, person_id: str, frame: int) -> str:
    kind = media_kind(question_type, person_id)
    if kind == "video":
        who = "all" if question_type == "temporal_social" else f"p{person_id}"
        return f"videos/{clip}/{clip}_{who}.mp4"
    suffix = {"individual": f"p{person_id}", "scene": "all", "plain": "plain"}[kind]
    return f"images/{clip}/{clip}_{int(frame):06d}_{suffix}.jpg"
