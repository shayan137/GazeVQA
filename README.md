<p align="center">
  <img src="figs/tasks.png" width="100%" alt="G3VQA: question types and their share of the benchmark">
</p>

<h1 align="center">Attention in Motion: G<sup>3</sup>VQA for<br/>Social Gaze Understanding in Video</h1>

<p align="center">
  <a href="https://shayan137.github.io/GazeVQA/">Project page</a> ·
  Paper (coming soon) ·
  Data (coming soon)
</p>

**G<sup>3</sup>VQA** (Group Glance Gaze VQA) is a video benchmark for social and temporal gaze reasoning. It asks a vision language model who is looking, what they look at, how attention is shared between people, and how all of this changes over a clip.

* 381 video clips from [ChildPlay](https://www.idiap.ch/en/scientific-research/data/childplay-gaze), with around 7K manually annotated key frames
* around 900K frame level question answer pairs in nine question types
* around 30K temporal questions in 21 templates that ask when, how often and in what order people look at objects and at each other, also while a face is hidden
* every gaze, appearance and social attention label was made and checked by human annotators, and no language model wrote a question or an answer
* a training free gaze tracking agent that gives the VLMs their gaze evidence and needs 83% less inference time than running the gaze estimator on every frame

We tested 15 VLMs. The best one answers 41% of the questions correctly, and accuracy drops from 32% on frame level questions to 21% on temporal ones. The [project page](https://shayan137.github.io/GazeVQA/) has ten test videos with the gaze annotations and temporal questions drawn on them, and the numbers are in the paper.

## Questions

Frame level questions are asked on one image:

* **Gaze point and direction:** gaze point, gaze direction
* **Attended object:** gaze object description, counting, absent object, ambiguous description
* **Social gaze:** shared gaze, exchanged gaze, group engagement

Temporal questions are asked on the whole clip. They are grouped into *when* (the answer is a frame number), *count* (how many times) and *memory* (which object or person, in which order, whether a target is revisited).

The benchmark is built on ChildPlay. We first corrected the gaze and head box labels of 50 ChildPlay videos and picked the key frames. The annotators then wrote the questions for each key frame. These are carried to the frames in between and reviewed again.

## How a model is tested

1. **Gaze tracking.** The agent estimates where every person looks. An image gaze estimator ([PaGE](https://arxiv.org/abs/2607.04860) in our runs, [Gaze-LLE](https://github.com/fkryan/gazelle) and [GazeAnywhere](https://github.com/IrohXu/GazeAnywhere) also work) runs only on the first frame of a person, when [OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) sees the head or eyes move by more than 2 degrees, or when the tracked point drifts more than one head diagonal. In between, Lucas Kanade optical flow follows the gaze point.
2. **Grounding.** For the attended object, a detector (RF-DETR, YOLO26 or WeDetect) proposes boxes and the box that fits the predicted gaze best is chosen, with the grounding method of [Gaze Target Agent](https://github.com/shayan137/Gaze-Target-Agent) ([paper](https://arxiv.org/abs/2609.06208)).
3. **Question answering.** The predicted gaze is drawn as a blue arrow from the head of each person to the gaze point. The VLM gets the image, or a video of at most 64 frames with the frame number printed in the corner, and the question. Scene questions show the arrows of all tracked people, questions about one person show one arrow, and ambiguous descriptions show none.
4. **Scoring.** Questions whose answer is yes or no, a count or a frame are scored by exact match on the verdict that is found in the free text answer. These are about 28% of all questions. All other answers are judged by [AnyJev](https://github.com/nokia-applied-research/AnyJev), a decision model that returns the probability that an answer is correct in one forward pass. An answer counts as correct at a probability of 0.5 or more. Accuracy is reported for the groups Individual, Ambiguous and Group (frame level), When, Count and Memory (temporal) and for all questions together.

## Installation

```bash
git clone https://github.com/shayan137/GazeVQA.git
cd GazeVQA
pip install -r requirements.txt
```

Gaze-LLE, GazeAnywhere and the other Hugging Face models are downloaded into `model_cache_dir` on first use. You also need:

* [OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) and its multitask checkpoint (`stage2_epoch_7_loss_1.1606_acc_0.5589.pth`), set in `configs/tracking.yaml`
* [vLLM](https://github.com/vllm-project/vllm) to serve the VLMs and the judge
* a separate Python environment for PaGE, made by `benchmarks/gaze_estimators/page/setup_env.sh`
* for GazeAnywhere only, a clone of its [repository](https://github.com/IrohXu/GazeAnywhere)

## Data

1. Download the ChildPlay videos and `clips.csv` and extract the frames:

```bash
python scripts/extract_frames.py --clips-csv /path/to/ChildPlay/clips.csv \
  --videos-dir /path/to/ChildPlay/videos --frames-dir data/frames --labels-dir data/labels
```

2. Download the G<sup>3</sup>VQA annotations (link coming soon) and arrange everything like this:

```text
data/
  frames/<clip>/<clip>_000001.jpg        # clip relative frame numbers, starting at 1
  labels/<clip>.csv                      # head boxes, gaze points, object boxes, appearance ("look like")
  annotations/test/<clip>_VQA.csv        # frame level questions and reference answers
  annotations/test/<clip>_temporal.csv   # temporal questions and reference answers
```

## Usage

### 1. Gaze tracking

PaGE runs in its own environment, so its predictions are computed once for every person in every frame and the agent replays them. The agent then counts only the frames on which it would have called the estimator.

```bash
python scripts/export_frame_list.py --labels-dir data/labels --frames-dir data/frames \
  --out outputs/gaze_benchmark/frame_list.csv

bash benchmarks/gaze_estimators/page/setup_env.sh
gaze_bench_envs/page/bin/python benchmarks/gaze_estimators/page/export_dense.py \
  --frame-list outputs/gaze_benchmark/frame_list.csv --out outputs/gaze_benchmark/caches/page.npz

python scripts/run_tracking.py --config configs/tracking.yaml
python scripts/run_tracking.py --config configs/tracking.yaml --dense --output-csv outputs/tracking/page_dense.csv
python scripts/eval_tracking.py outputs/tracking/page_dense.csv outputs/tracking/page_tracking.csv
```

`eval_tracking.py` prints the number of estimator calls, AUC, average L2 distance and in or out AP. To run Gaze-LLE or GazeAnywhere directly, use `--estimator gazelle` or `--estimator gaze_anywhere`. The folders of Sharingan and Gupta et al. in `benchmarks/gaze_estimators/` work the same way as the PaGE one.

### 2. Grounding

```bash
python benchmarks/grounding/run_detect.py --detector rfdetr_2xl       # or yolo26x, wedetect_large_uni
python benchmarks/grounding/select_gaze_target.py --detector rfdetr_2xl
python benchmarks/grounding/eval_grounding.py --ann-dir data/labels
```

The last step prints mIoU, Acc@50, Acc@75 and AP@50 against the annotated object boxes.

### 3. Question answering

Render the images and videos that the models see once, so that the answer jobs only read files:

```bash
python scripts/render_vqa_inputs.py --questions-dir data/annotations/test \
  --predictions-csv outputs/tracking/page_tracking.csv --frames-dir data/frames \
  --out-dir outputs/vqa_inputs --images --videos
```

Serve a model with vLLM and ask it every question:

```bash
vllm serve Qwen/Qwen3.5-27B --served-model-name qwen3.5-27b --port 8000 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 1, "video": 1}' --media-io-kwargs '{"video": {"num_frames": -1}}' --trust-remote-code

python scripts/run_vqa.py --config configs/vqa.yaml --answer-model qwen3.5-27b --output-csv outputs/vqa/qwen3.5-27b.csv
```

Every question, reference answer, model answer and what was sent to the model is written to one CSV. `--resume` continues an interrupted run, and `slurm/run_vqa.sbatch` does both steps on a SLURM node.

### 4. Scores

Serve the judge, score the free text answers, and combine them with the exact matches:

```bash
vllm serve morriszjm/Tacit-9B --port 8100 --max-model-len 16384 --limit-mm-per-prompt '{"image": 0, "video": 0}'
python scripts/judge_answers.py --server http://localhost:8100 --model-path morriszjm/Tacit-9B \
  --answers-dir outputs/vqa --out-dir outputs/judge

python scripts/score_vqa.py --answers-dir outputs/vqa --judge-dir outputs/judge --out outputs/scores.csv
```

`slurm/judge_answers.sbatch` does the first two commands on one node. `score_vqa.py` prints one row per model with the accuracy of every group and the share of answers that were scored by exact match.

## Code structure

```text
source/
  config.py                  # TrackingConfig and VQAConfig (YAML)
  data/
    childplay.py             # per clip label CSVs
    vqa.py                   # question files and question types
  tracking/
    agent.py                 # the tracking agent: estimator on trigger frames, optical flow in between
    estimators.py            # Gaze-LLE, GazeAnywhere and the replay of cached predictions (PaGE)
    openface.py              # head and eye direction from OpenFace 3.0
    triggers.py              # direction changes become trigger frames
    point_tracker.py         # Lucas Kanade point tracker
    metrics.py               # AUC, L2, in or out AP
  vqa/
    visual_prompt.py         # gaze arrows and the sentence before the question
    predictions.py           # gaze predictions for each question
    media.py, video.py       # which rendered image or video belongs to a question
    backend.py               # vLLM client
    runner.py                # concurrent answering
    closed_form.py           # exact match of yes or no, counts and frames
scripts/
  extract_frames.py, export_frame_list.py, run_tracking.py, eval_tracking.py,
  render_vqa_inputs.py, run_vqa.py, judge_answers.py, score_vqa.py
benchmarks/
  gaze_estimators/           # dense prediction export for PaGE, Gaze-LLE, GazeAnywhere, Sharingan, Gupta et al.
  grounding/                 # detectors, gaze target selection and grounding metrics
configs/                     # tracking.yaml, vqa.yaml
slurm/                       # run_vqa.sbatch, judge_answers.sbatch
docs/                        # the project page (GitHub Pages)
```

## Citation

The paper is under review. A BibTeX entry will be added here after publication.

## Acknowledgements

G<sup>3</sup>VQA builds on [ChildPlay](https://www.idiap.ch/en/scientific-research/data/childplay-gaze). We use [PaGE](https://arxiv.org/abs/2607.04860), [Gaze-LLE](https://github.com/fkryan/gazelle), [GazeAnywhere](https://github.com/IrohXu/GazeAnywhere), [OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0) and [AnyJev](https://github.com/nokia-applied-research/AnyJev), and we follow the grounding method of [Gaze Target Agent](https://github.com/shayan137/Gaze-Target-Agent).
