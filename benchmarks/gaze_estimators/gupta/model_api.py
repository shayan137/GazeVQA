"""Gupta, Tafasca, Odobez, CVPRW 2022, "A Modular Multimodal Architecture for
Gaze Target Prediction" (AttentionModelCombined: RGB + depth + pose
EfficientNet branches, human-centric gaze-cone branch): construction and
one-call inference mirroring eval_on_videoatttarget.py test() and
dataset.py VideoAttTarget_image (test path)."""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("GUPTA_REPO", "multimodal_gaze_target_prediction-main")
# the repo's config.py has human_centric_weights = ''; the shadow config.py in this folder must win
for p in (REPO, HERE):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.environ.setdefault("TORCH_HOME", os.path.join("model_cache", "torch_hub"))  # efficientnet_pytorch / torchvision weights

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

import config as gupta_config  # noqa: E402  (the shadow config)
from model import AttentionModelCombined  # noqa: E402
from utils import gazeutils, imutils  # noqa: E402

assert gupta_config.__file__.startswith(HERE), gupta_config.__file__

INPUT_RES = gupta_config.input_resolution       # 224
CONE_RES = INPUT_RES if gupta_config.cone_mode == "early" else INPUT_RES // 4  # dataset.py: cone_resolution
BOX_EXPAND_K = 0.1                              # dataset.py "expand face bbox a bit"
DEPTH_MAX = 65535                               # dataset.py: depth = depth / 65535
IMAGE_TRANSFORM = transforms.Compose([           # transforms._get_transform
    transforms.Resize((INPUT_RES, INPUT_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
MODALITY_TRANSFORM = transforms.Compose([        # transforms._get_transform_modality
    transforms.Resize((INPUT_RES, INPUT_RES)),
    transforms.ToTensor(),
])


def checkpoint_settings(state: dict, ck: dict) -> tuple[str, bool]:
    """cone_mode / pred_inout as eval_on_videoatttarget.py reads them from the
    checkpoint, falling back to the weights when the keys are absent (they are
    absent in the released attention-*.pt files): early fusion adds the gaze
    cone + head mask to the stem (5 input channels); pred_inout iff the
    in_vs_out_head weights exist."""
    cone_mode = ck.get("cone_mode")
    if cone_mode is None:
        cone_mode = "early" if state["feature_extractor_image.backbone._conv_stem.weight"].shape[1] == 5 else "late"
    pred_inout = ck.get("pred_inout")
    if pred_inout is None:
        pred_inout = any(k.startswith("in_vs_out_head.") for k in state)
    return cone_mode, bool(pred_inout)


_FPN_KEY = re.compile(r"^(.*\.fpn\.(?:inner|layer)_blocks\.\d+)\.(weight|bias)$")


def remap_fpn_keys(state: dict) -> dict:
    """The checkpoints were saved with torchvision 0.10, whose
    FeaturePyramidNetwork stored plain convs (fpn.inner_blocks.0.weight);
    torchvision >= 0.12 wraps them in Conv2dNormActivation
    (fpn.inner_blocks.0.0.weight). torchvision only converts the old names when
    the state dict carries version metadata, which the repo's
    model_dict.update() path bypasses, so the rename is done here."""
    return {(f"{m.group(1)}.0.{m.group(2)}" if (m := _FPN_KEY.match(k)) else k): v for k, v in state.items()}


def build(checkpoint: str, device: str):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ck.get("modality", "attention") == "attention", ck.get("modality")
    state = remap_fpn_keys(ck["model"])
    cone_mode, pred_inout = checkpoint_settings(state, ck)
    model = AttentionModelCombined(cone_mode=cone_mode, pred_inout=pred_inout)  # loads human_centric_weights + ImageNet EfficientNets
    model_dict = model.state_dict()
    missing = [k for k in model_dict if k not in state]
    unexpected = [k for k in state if k not in model_dict]
    model_dict.update(state)
    model.load_state_dict(model_dict, strict=True)  # eval_on_videoatttarget.py test()
    print(f"[gupta] cone_mode={cone_mode} pred_inout={pred_inout} backbone_name={ck.get('backbone_name')} epoch={ck.get('epoch')}: "
          f"{len(missing)} missing, {len(unexpected)} unexpected keys" + (f" {missing[:3]} {unexpected[:3]}" if missing or unexpected else " (strict match)"))
    model.cone_mode_used, model.pred_inout_used = cone_mode, pred_inout
    return model.to(device).eval()


def aux_paths(aux_dir: str, image_path: str) -> tuple[str, str]:
    clip = os.path.basename(os.path.dirname(image_path))
    base = os.path.splitext(os.path.basename(image_path))[0]
    return (os.path.join(aux_dir, "midas_dpt_large", clip, base + ".png"),
            os.path.join(aux_dir, "pose_keypointrcnn", clip, base + ".jpg"))


def load_depth(path: str) -> Image.Image:
    """16-bit MiDaS PNG. The authors' Pillow 8.4 opened it as mode 'I' (int32),
    which torchvision's ToTensor keeps as int32 so that /65535 lands in [0, 1];
    Pillow >= 10 opens it as 'I;16', which ToTensor reads as int16 (values above
    32767 wrap). Converting to 'I' reproduces the original pipeline."""
    im = Image.open(path)
    im.load()
    return im if im.mode == "I" else im.convert("I")


def load_pose(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def expand_box(box):
    """dataset.py: sequential expansion (x_max uses the already-moved x_min); no clipping on the test path."""
    x_min, y_min, x_max, y_max = map(float, box)
    k = BOX_EXPAND_K
    x_min -= k * abs(x_max - x_min)
    y_min -= k * abs(y_max - y_min)
    x_max += k * abs(x_max - x_min)
    y_max += k * abs(y_max - y_min)
    return x_min, y_min, x_max, y_max


def gaze_field_tensor(eye_xy_norm) -> torch.Tensor:
    """dataset.py: gazeutils.generate_data_field(eye, cone_res, cone_res), divided by its norm (min 0.1)."""
    field = gazeutils.generate_data_field(np.array(eye_xy_norm, dtype=np.float64), width=CONE_RES, height=CONE_RES)
    norm = np.sqrt(np.sum(field ** 2, axis=0)).reshape([1, CONE_RES, CONE_RES])
    norm = np.maximum(norm, 0.1)
    field /= norm
    return torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))


def build_inputs(image, depth_i, pose_rgb, box, device):
    """Frame, depth, pose, face crop, gaze field and head mask as in VideoAttTarget_image (test)."""
    W, H = image.size
    x_min, y_min, x_max, y_max = map(float, box)
    eye = ((x_min + x_max) / 2.0 / W, (0.65 * y_min + 0.35 * y_max) / H)  # from the un-expanded box
    ex1, ey1, ex2, ey2 = expand_box(box)
    head_t = imutils.get_head_box_channel(ex1, ey1, ex2, ey2, W, H, resolution=CONE_RES, coordconv=False).unsqueeze(0)
    face = image.crop((int(ex1), int(ey1), int(ex2), int(ey2)))  # PIL pads black outside the frame, as in the repo
    img_t = IMAGE_TRANSFORM(image)
    pose_t = MODALITY_TRANSFORM(pose_rgb)
    depth_t = MODALITY_TRANSFORM(depth_i) / DEPTH_MAX
    face_t = IMAGE_TRANSFORM(face)
    field_t = gaze_field_tensor(eye)
    ts = [t.unsqueeze(0).to(device).float() for t in (img_t, depth_t, pose_t, face_t, field_t, head_t)]
    return (*ts, (int(ex1), int(ey1), int(ex2), int(ey2)))


def forward(model, img_t, depth_t, pose_t, face_t, field_t, head_t):
    hm, direction, inout, att = model([img_t, depth_t, pose_t], face_t, field_t, head_t)
    return hm, direction, inout, att


def predict(model, image, depth_i, pose_rgb, box, device):
    img_t, depth_t, pose_t, face_t, field_t, head_t, pbox = build_inputs(image, depth_i, pose_rgb, box, device)
    with torch.no_grad():
        hm, _direction, inout, _att = forward(model, img_t, depth_t, pose_t, face_t, field_t, head_t)
    # InvsOut ends with nn.Sigmoid(), and eval_on_videoatttarget.py uses the output directly as the in/out score
    p_in = float(inout.reshape(-1)[0].item()) if getattr(model, "pred_inout_used", True) else float("nan")
    return hm[0, 0].float().cpu().numpy(), p_in, pbox
