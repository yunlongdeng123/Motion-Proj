#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/drivingeditor_repro
mkdir -p "$BASE/weights"
download() {
  name="$1"
  url="$2"
  if [ ! -s "$BASE/weights/$name" ]; then
    curl --fail --location --retry 3 --connect-timeout 20 --continue-at - --output "$BASE/weights/$name.partial" "$url"
    mv "$BASE/weights/$name.partial" "$BASE/weights/$name"
  fi
  sha256sum "$BASE/weights/$name"
}
trap 'code=$?; printf "%s\n" "$code" > "$BASE/logs/mask_weights.exit"' EXIT
download DEVA-propagation.pth https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
download sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
download groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
download mobile_sam.pt https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt
download GroundingDINO_SwinT_OGC.py https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py
sha256sum "$BASE/weights/"* > "$BASE/logs/mask_weights.sha256"
