"""Text-only LLM judge.

The judge sees the question, the reference answer(s) and the candidate
answer, but not the image, so it compares against the reference instead of
solving the question itself.
"""

import re

JUDGE_PROMPT = """You are grading a candidate answer to a question against reference answer(s), for a question about where someone in a photo is looking.

Question: {question}
Reference answer(s) -- matching any one counts as correct: {reference_answers}
Candidate answer to grade: {candidate_answer}

Score the candidate answer's correctness against the reference answer(s) on a 0-5 scale:
5 = fully correct, matches a reference answer's meaning
3 = partially correct, right idea but missing or wrong detail
0 = wrong or unrelated

Respond with only the integer score, nothing else."""

_SCORE_RE = re.compile(r"[0-5]")


def build_judge_prompt(question: str, reference_answers: str, candidate_answer: str) -> str:
    return JUDGE_PROMPT.format(question=question, reference_answers=reference_answers, candidate_answer=candidate_answer)


def parse_judge_score(text: str) -> int | None:
    # the last digit, in case the judge writes something before the score
    matches = _SCORE_RE.findall(text.strip())
    return int(matches[-1]) if matches else None
