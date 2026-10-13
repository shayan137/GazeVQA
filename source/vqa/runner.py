"""Answer, judge and score every question, with many requests in flight.

Results are appended to one CSV (question, references, answer, judge score,
embedding similarity) every `flush_every` rows, so an interrupted run can be
resumed.
"""

import asyncio
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from source.vqa.judge import build_judge_prompt, parse_judge_score
from source.vqa.metrics import embedding_similarity
from source.vqa.visual_prompt import build_overlay, build_prompt

OUTPUT_COLUMNS = [
    "clip", "person_id", "frame", "vqa_id", "question_type", "question", "reference_answers",
    "n_arrows", "image_path", "has_overlay",
    "answer_model", "vlm_answer", "vlm_latency_s", "vlm_error",
    "judge_model", "judge_score", "judge_raw", "judge_error",
    "embedding_similarity", "skipped_reason",
]


async def judge_answer(judge, row, answer: str, max_new_tokens: int) -> tuple:
    prompt = build_judge_prompt(row["question"], row["reference_answers"], answer)
    try:
        raw = await judge.generate(None, prompt, max_new_tokens=max_new_tokens)
        return parse_judge_score(raw), raw, ""
    except Exception as e:
        return None, "", repr(e)


async def answer_row(row, predictions, answerer, judge, embed_model, max_new_tokens: int, judge_max_new_tokens: int) -> dict:
    image_path, arrows, has_overlay, is_scene = predictions.lookup(row["clip"], row["person_id"], row["frame"])
    result = {
        **{k: row[k] for k in ("clip", "person_id", "frame", "vqa_id", "question_type", "question", "reference_answers")},
        "n_arrows": len(arrows), "image_path": image_path or "", "has_overlay": has_overlay,
        "answer_model": answerer.model, "judge_model": judge.model if judge else "",
        "vlm_answer": "", "vlm_latency_s": 0.0, "vlm_error": "",
        "judge_score": None, "judge_raw": "", "judge_error": "",
        "embedding_similarity": float("nan"), "skipped_reason": "",
    }
    if image_path is None:
        result["skipped_reason"] = "no_gaze_point"
        return result

    if has_overlay:
        image = await asyncio.to_thread(build_overlay, image_path, arrows)
    else:
        image = await asyncio.to_thread(lambda: Image.open(image_path).convert("RGB"))
    prompt = build_prompt(row["question"], is_scene, has_overlay)

    t0 = time.monotonic()
    try:
        result["vlm_answer"] = await answerer.generate(image, prompt, max_new_tokens=max_new_tokens)
    except Exception as e:
        result["vlm_error"] = repr(e)
    result["vlm_latency_s"] = time.monotonic() - t0

    answer = result["vlm_answer"]
    if answer:
        if judge is not None:
            result["judge_score"], result["judge_raw"], result["judge_error"] = await judge_answer(
                judge, row, answer, judge_max_new_tokens
            )
        result["embedding_similarity"] = await asyncio.to_thread(
            embedding_similarity, answer, row["reference_answers"], embed_model
        )
    return result


async def rejudge_row(row, judge, embed_model, judge_max_new_tokens: int) -> dict:
    result = dict(row)
    answer = result.get("vlm_answer")
    if not answer or pd.isna(answer):
        return result
    result["judge_score"], result["judge_raw"], result["judge_error"] = await judge_answer(
        judge, row, answer, judge_max_new_tokens
    )
    result["judge_model"] = judge.model
    result["embedding_similarity"] = await asyncio.to_thread(
        embedding_similarity, answer, result["reference_answers"], embed_model
    )
    return result


async def _run_bounded(rows: list[dict], worker, output_csv: Path, append: bool, max_concurrency: int, flush_every: int = 50):
    sem = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()
    buffer = []
    state = {"first": not append, "done": 0}

    def flush():
        first = state["first"]
        pd.DataFrame(buffer, columns=OUTPUT_COLUMNS).to_csv(output_csv, mode="w" if first else "a", header=first, index=False)
        state["first"] = False
        buffer.clear()

    async def bounded(row):
        async with sem:
            result = await worker(row)
        async with lock:
            buffer.append(result)
            state["done"] += 1
            if state["done"] % 500 == 0:
                print(f"[{state['done']}/{len(rows)}]", flush=True)
            if len(buffer) >= flush_every:
                flush()

    await asyncio.gather(*(bounded(r) for r in rows))
    if buffer:
        flush()


def _done_keys(output_csv: Path) -> set:
    done = pd.read_csv(output_csv, usecols=["clip", "person_id", "frame", "vqa_id"], dtype={"person_id": str})
    return set(done.itertuples(index=False, name=None))


async def run_vqa(questions: pd.DataFrame, predictions, answerer, judge, embed_model, output_csv, resume: bool = False,
                  max_concurrency: int = 128, max_new_tokens: int = 4096, judge_max_new_tokens: int = 4096) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    append = resume and output_csv.exists()
    done = _done_keys(output_csv) if append else set()
    rows = [r for r in questions.to_dict("records") if (r["clip"], r["person_id"], r["frame"], r["vqa_id"]) not in done]
    print(f"{len(rows)} questions to answer ({len(done)} already done)", flush=True)

    async def worker(row):
        return await answer_row(row, predictions, answerer, judge, embed_model, max_new_tokens, judge_max_new_tokens)

    await _run_bounded(rows, worker, output_csv, append, max_concurrency)
    await answerer.close()
    if judge is not None:
        await judge.close()


async def rejudge(input_csv, judge, embed_model, output_csv, max_concurrency: int = 128, judge_max_new_tokens: int = 4096) -> None:
    output_csv = Path(output_csv)
    if output_csv.resolve() == Path(input_csv).resolve():
        raise ValueError("write the rejudged results to a new file, not over the input")
    rows = pd.read_csv(input_csv, dtype={"person_id": str}).to_dict("records")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    async def worker(row):
        return await rejudge_row(row, judge, embed_model, judge_max_new_tokens)

    await _run_bounded(rows, worker, output_csv, False, max_concurrency)
    await judge.close()
