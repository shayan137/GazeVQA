"""Gaze-guided visual prompt: a blue arrow from each relevant person's head
to their predicted gaze point, and a one-line explanation before the question.
"""

import math

import cv2
from PIL import Image

ARROW_COLOR_BGR = (255, 0, 0)
ARROW_WIDTH_PX = 4

Point = tuple[float, float]


def head_center(x1: float, y1: float, x2: float, y2: float) -> Point:
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def draw_gaze_arrow(frame, p0: Point, p1: Point, color=ARROW_COLOR_BGR, lw: int = ARROW_WIDTH_PX) -> None:
    x0, y0, x1, y1 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 8:
        return
    tip = max(0.10, min(0.35, 22 / length))
    cv2.arrowedLine(frame, (x0, y0), (x1, y1), color, lw, tipLength=tip, line_type=cv2.LINE_AA)
    cv2.circle(frame, (x1, y1), max(3, lw + 2), color, -1, lineType=cv2.LINE_AA)


def build_overlay(image_path, arrows: list[tuple[Point, Point]]) -> Image.Image:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(image_path)
    for head, gaze in arrows:
        draw_gaze_arrow(frame, head, gaze)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def build_prompt(question: str, is_scene: bool, has_overlay: bool = True) -> str:
    if not has_overlay:
        return question
    if is_scene:
        return f"The blue arrows show where the people are looking. {question}"
    return f"The blue arrow shows where the person looks. {question}"
