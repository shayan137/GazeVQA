#!/usr/bin/env python3
"""
Judge the free-text answers of a VLM with AnyJev (https://github.com/nokia-applied-research/AnyJev).

AnyJev is a decision model. It reads the question, the reference answer(s) and the candidate answer and returns the probability that
the candidate is correct in a single forward pass, so no explanation has to be generated, which makes it much faster than a normal LLM judge.

  # serve the judge once (any GPU with enough memory for a 9B model)
  vllm serve morriszjm/Tacit-9B --port 8100 --max-model-len 16384 --limit-mm-per-prompt '{"image": 0, "video": 0}'

  python scripts/judge_answers.py --server http://localhost:8100 --model-path morriszjm/Tacit-9B \
      --answers-dir outputs/vqa --out-dir outputs/judge

--answers-dir holds one answers CSV per model (the output of scripts/run_vqa.py). For each file <model>.csv the script writes
<out-dir>/<model>.csv with the columns clip, person_id, frame, vqa_id, p_correct. A run can be stopped and started again, rows that are
already in the output file are skipped. The judge never sees the image or the video, and not the name of the model that wrote the answer.
Answers longer than 1,500 characters keep their first and last 750, because the verdict is almost always at one end.
"""
import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

KEYS = ["clip", "person_id", "frame", "vqa_id"]
SEP = " ||| "
QUESTION = ("Compare the candidate answer with the GT answer. The candidate answer is correct when it gives the same person, object, count, "
            "frame or yes or no verdict as the GT answer, even if it is worded differently. Is the candidate answer correct?")


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def shorten(text, limit=1500):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit // 2] + " ... " + text[-limit // 2:]


def build_state(question, reference_answers, answer):
    refs = " | ".join(x for x in str(reference_answers).split(SEP) if x)
    return f"Question: {question}\nGT answer(s): {refs}\nCandidate answer: {shorten(answer)}\n"


def p_yes(output):
    for label, prob in output["probs"].items():
        if str(label).strip().lower() == "yes":
            return float(prob)
    raise KeyError(f"no 'yes' in {output['probs']}")


def load_judge(args):
    from anyjev import Tacit

    tacit = Tacit.from_pretrained(args.model_path, engine="server", base_url=args.server, api_key="none")
    tacit.engine.timeout = args.request_timeout
    return tacit


def judge_model(args, tacit, model):
    src = os.path.join(args.answers_dir, f"{model}.csv")
    dst = os.path.join(args.out_dir, f"{model}.csv")
    df = pd.read_csv(src, dtype=str, keep_default_na=False, usecols=KEYS + ["question", "reference_answers", "vlm_answer", "skipped_reason"])
    df = df[df["skipped_reason"] == ""].drop_duplicates(subset=KEYS, keep="last").reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    done = set()
    if os.path.exists(dst):
        with open(dst, "rb+") as fh:  # a killed run can leave the last line unfinished, close it so the next row starts clean
            fh.seek(0, 2)
            if fh.tell() and (fh.seek(-1, 2) or fh.read(1)) != b"\n":
                fh.write(b"\n")
        old = pd.read_csv(dst, dtype=str, keep_default_na=False, on_bad_lines="skip")
        old = old[old["p_correct"].str.match(r"^[0-9.eE+-]+$", na=False)]
        done = set(map(tuple, old[KEYS].values.tolist()))
    todo = df[[tuple(k) not in done for k in df[KEYS].values.tolist()]]
    log(f"{model}: {len(df)} answered questions, {len(done)} judged before, {len(todo)} to do")
    if not len(todo):
        return

    new_file = not os.path.exists(dst)
    fh = open(dst, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(KEYS + ["p_correct"])
    lock = threading.Lock()
    count = {"done": 0, "failed": 0}
    t0 = time.time()

    def work(chunk):
        keys = chunk[KEYS].values.tolist()
        empty = {i for i, ans in enumerate(chunk["vlm_answer"]) if not str(ans).strip()}
        ask = [i for i in range(len(chunk)) if i not in empty]
        probs = {i: 0.0 for i in empty}  # an empty answer is wrong
        if ask:
            items = [dict(state=build_state(chunk["question"].iloc[i], chunk["reference_answers"].iloc[i], chunk["vlm_answer"].iloc[i]),
                          question=QUESTION, kind="yes_no") for i in ask]
            for attempt in range(4):
                try:
                    for i, out in zip(ask, tacit.decide_batch(items)):
                        probs[i] = p_yes(out)
                    break
                except Exception as e:  # noqa: BLE001  (the server may be busy: wait and try again)
                    log(f"  request failed (try {attempt + 1}): {str(e)[:200]}")
                    time.sleep(10 * (attempt + 1))
        with lock:
            for i, key in enumerate(keys):
                if i in probs:
                    writer.writerow(key + [f"{probs[i]:.6f}"])
                else:
                    count["failed"] += 1
            fh.flush()
            count["done"] += len(keys)
            if count["done"] % (args.chunk * 40) < args.chunk:
                rate = count["done"] / (time.time() - t0)
                log(f"  {model}: {count['done']}/{len(todo)} ({rate:.0f} per second)")

    chunks = [todo.iloc[i:i + args.chunk] for i in range(0, len(todo), args.chunk)]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, chunks))
    fh.close()
    log(f"{model}: finished in {time.time() - t0:.0f}s, {count['failed']} questions without a score")
    if count["failed"] > 0.001 * len(todo):
        sys.exit(f"{model}: too many questions failed ({count['failed']}), run the script again to continue")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="URL of the running vllm server of the judge")
    ap.add_argument("--model-path", required=True, help="the judge model, as it was given to vllm serve")
    ap.add_argument("--answers-dir", required=True, help="folder with one answers CSV per model")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", default="", help="comma-separated model names, default all CSVs in --answers-dir")
    ap.add_argument("--chunk", type=int, default=64, help="questions per request")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N questions of each model (quick test)")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m] or sorted(f[:-4] for f in os.listdir(args.answers_dir) if f.endswith(".csv"))
    os.makedirs(args.out_dir, exist_ok=True)
    tacit = load_judge(args)
    for model in models:
        judge_model(args, tacit, model)
    log("done")


if __name__ == "__main__":
    main()
