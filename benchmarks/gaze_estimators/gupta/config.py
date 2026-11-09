"""Shadow of the Gupta et al. (CVPRW 2022) repo's config.py, placed first on
sys.path by model_api.py so that `from config import ...` inside the repo's
model.py / dataset.py resolves here. Same values as the repo's config.py (VAT
setting: pred_inout=True, cone_mode='early') except human_centric_weights,
which is empty in the repo and must point at the released human-centric
checkpoint (AttentionModelCombined.__init__ loads it)."""
import os

# =============================================================================
# GazeFollow dataset dir config (unused by the adapter)
# =============================================================================
gazefollow_data = "data/gazefollow"
gazefollow_pose = "data/gazefollow-pose"
gazefollow_depth = "data/gazefollow-depth"
gazefollow_train_label = "data/gazefollow/train_annotations_release.txt"
gazefollow_val_label = "data/gazefollow/test_annotations_release.txt"

# =============================================================================
# VideoAttTarget dataset dir config (unused by the adapter)
# =============================================================================
videoattentiontarget_data = "data/videoatttarget/images"
videoattentiontarget_pose = "data/videoatttarget-pose/images"
videoattentiontarget_depth = "data/videoatttarget-depth/images"
videoattentiontarget_train_label = "data/videoatttarget/annotations/train"
videoattentiontarget_val_label = "data/videoatttarget/annotations/test"

# path to pretrained weights for human-centric branch
human_centric_weights = os.environ.get("GUPTA_HUMAN_CENTRIC_WEIGHTS",
                                       os.path.join(os.environ.get("GAZE_CKPTS", "gaze_ckpts"), "gupta", "human-centric.pt"))

# =============================================================================
# model config
# =============================================================================
input_resolution = 224
output_resolution = 64

cone_mode = 'early'    # {'late', 'early'} fusion of person information
modality_dropout = True    # only used for attention model
pred_inout = True    # {set True for VideoAttentionTarget}
privacy = False     # {set True to train/test privacy-sensitive model}

# pytorch amp to speed up training and reduce memory usage
use_amp = False
