#!/usr/bin/env bash
# Sharingan (CVPR 2024) checkpoints from Zenodo record 14066123 (sharingan_checkpoints.tar.gz, 1.2 GB:
# gazefollow.pt, videoattentiontarget.pt, childplay.pt). Idempotent: skips the download / extraction if present.
set -euo pipefail
DEST=${GAZE_CKPTS:-gaze_ckpts}/sharingan
mkdir -p "$DEST"
TAR="$DEST/sharingan_checkpoints.tar.gz"
if [ ! -f "$DEST/checkpoints/childplay.pt" ] || [ ! -f "$DEST/checkpoints/videoattentiontarget.pt" ]; then
  if [ ! -f "$TAR" ]; then
    wget -nv -c -O "$TAR.part" "https://zenodo.org/api/records/14066123/files/sharingan_checkpoints.tar.gz/content"
    mv "$TAR.part" "$TAR"
  fi
  tar -tzf "$TAR" > "$DEST/sharingan_checkpoints.contents.txt"
  mkdir -p "$DEST/checkpoints"
  # the archive holds the .pt files (possibly inside a folder); flatten them into $DEST/checkpoints/
  tar -xzf "$TAR" -C "$DEST/checkpoints" --no-same-owner --strip-components="$(awk -F/ 'NF>max{max=NF} END{print max-1}' "$DEST/sharingan_checkpoints.contents.txt")" --wildcards '*.pt'
fi
ls -la "$DEST/checkpoints"
