"""YAML configuration for the gaze tracking agent and the VQA evaluation.

Every field can be set in the YAML file; the scripts also expose the
commonly changed ones (paths, model names) as command-line overrides.
"""

import dataclasses

import yaml


def load_yaml(cls, path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return cls(**data)


@dataclasses.dataclass
class TrackingConfig:
    labels_dir: str  # per-clip label CSVs (head boxes, gaze points, appearance)
    frames_dir: str  # extracted frames, <clip>/<clip>_<frame:06d>.jpg
    output_csv: str

    estimator: str = "gazelle"  # gazelle | gaze_anywhere | cached
    dense: bool = False  # run the estimator on every frame (baseline)
    device: str = "cuda"

    # One shared folder every auto-downloaded model caches into.
    model_cache_dir: str = "model_cache"

    gazelle_repo: str = "fkryan/gazelle"
    gazelle_model: str = "gazelle_dinov2_vitl14_inout"
    gaze_anywhere_model: str = "IrohXu/GazeAnywhere"
    gaze_anywhere_repo: str = ""  # local clone of github.com/IrohXu/GazeAnywhere

    # estimator: cached -> dense prediction cache (.npz) written by
    # benchmarks/gaze_estimators/<model>/export_dense.py
    cache_path: str = ""

    # OpenFace 3.0 multitask model, used to detect head/eye movement
    openface_root: str = ""
    openface_checkpoint: str = ""
    # yaw/pitch per labelled head crop from an earlier run (head_pose output);
    # OpenFace only runs for person-frames missing from it
    openface_cache_csv: str = ""

    # copy each track's appearance description ("look like", annotated on one
    # frame) to all of its frames, for text-prompted estimators
    propagate_look_like: bool = False

    trigger_threshold_deg: float = 2.0
    min_peak_distance: int = 3
    drift_head_diagonals: float = 1.0
    inout_threshold: float = 0.35

    @classmethod
    def from_yaml(cls, path: str) -> "TrackingConfig":
        return load_yaml(cls, path)


@dataclasses.dataclass
class VQAConfig:
    vqa_dir: str  # per-clip G3VQA question files, <clip>.json (question_format json) or csv/test of an annotation package (csv)
    predictions_csv: str  # output of scripts/run_tracking.py
    frames_dir: str
    output_csv: str

    # The answering model is served with vLLM's OpenAI-compatible server.
    answer_url: str = "http://localhost:8000/v1"
    answer_model: str = ""

    question_format: str = "json"  # json: <clip>.json files; csv: <clip>_VQA.csv + <clip>_temporal.csv
    extended: bool = False  # also ask group-engagement and the whole-video temporal questions
    media_dir: str = ""  # output of scripts/render_vqa_inputs.py; missing images are drawn on the fly
    video_transport: str = "data"  # data: mp4 inside the request, file: path read by the server (--allowed-local-media-path)

    model_cache_dir: str = "model_cache"

    max_concurrency: int = 128
    max_new_tokens: int = 4096
    request_timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: str) -> "VQAConfig":
        return load_yaml(cls, path)
