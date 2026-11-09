#!/usr/bin/env bash
# Gupta et al., CVPRW 2022 checkpoints (SWITCHdrive links from the repo README):
#   human-centric.pt              gaze-cone / direction branch (loaded by AttentionModelCombined.__init__)
#   attention-videoatttarget.pt   attention model fine-tuned on VideoAttentionTarget (in/out head)
#   attention-gazefollow.pt       attention model trained on GazeFollow (no in/out head)
# The /download endpoints answer with a Content-Disposition attachment carrying these names.
# Idempotent: skips files that exist and are larger than 50 MB. ImageNet EfficientNet-b0/b1 weights
# (efficientnet_pytorch) and the KeypointRCNN weights are fetched by torch.hub into TORCH_HOME on first use.
set -euo pipefail
D=${GAZE_CKPTS:-gaze_ckpts}/gupta
mkdir -p "$D"; cd "$D"
fetch() {  # name url
  if [ -f "$1" ] && [ "$(stat -c %s "$1")" -gt 50000000 ]; then return; fi
  curl -sSL -o "$1.part" "$2"
  if file "$1.part" | grep -qi "html"; then echo "$2 returned HTML, not a checkpoint"; head -c 400 "$1.part"; exit 1; fi
  mv "$1.part" "$1"
}
fetch human-centric.pt            "https://drive.switch.ch/index.php/s/5hDsBdP4OsLks5X/download"
fetch attention-videoatttarget.pt "https://drive.switch.ch/index.php/s/EjVQlvUDisvL1c4/download"
fetch attention-gazefollow.pt     "https://drive.switch.ch/index.php/s/fJVjWSJWQtoJeT3/download"
ls -la "$D"
