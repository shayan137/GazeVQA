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

    estimator: str = "gazelle"  # gazelle | gaze_anywhere
    dense: bool = False  # run the estimator on every frame (baseline)
    device: str = "cuda"

    # One shared folder every auto-downloaded model caches into.
    model_cache_dir: str = "model_cache"

    gazelle_repo: str = "fkryan/gazelle"
    gazelle_model: str = "gazelle_dinov2_vitl14_inout"
    gaze_anywhere_model: str = "IrohXu/GazeAnywhere"
    gaze_anywhere_repo: str = ""  # local clone of github.com/IrohXu/GazeAnywhere

    # OpenFace 3.0 multitask model, used to detect head/eye movement
    openface_root: str = ""
    openface_checkpoint: str = ""

    trigger_threshold_deg: float = 2.0
    min_peak_distance: int = 3
    drift_head_diagonals: float = 1.0
    inout_threshold: float = 0.35

    @classmethod
    def from_yaml(cls, path: str) -> "TrackingConfig":
        return load_yaml(cls, path)


@dataclasses.dataclass
class VQAConfig:
    vqa_dir: str  # per-clip G3VQA question files, <clip>.json
    predictions_csv: str  # output of scripts/run_tracking.py
    frames_dir: str
    output_csv: str

    # Both models are served with vLLM's OpenAI-compatible server.
    answer_url: str = "http://localhost:8000/v1"
    answer_model: str = ""
    judge_url: str = "http://localhost:8001/v1"
    judge_model: str = "qwen3.8-27b"  # empty -> answer only, judge later with rejudge_vqa.py

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: str = "model_cache"

    max_concurrency: int = 128
    max_new_tokens: int = 4096
    judge_max_new_tokens: int = 4096
    request_timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: str) -> "VQAConfig":
        return load_yaml(cls, path)
