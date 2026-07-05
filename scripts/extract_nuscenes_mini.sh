#!/usr/bin/env bash
# Extract the nuScenes mini split (read-only shared archive) onto the data disk.
# Mini is ~4GB compressed and extracts to a `samples/`, `sweeps/`, `maps/`,
# `v1.0-mini/` layout that nuscenes-devkit expects under a single dataroot.
set -euo pipefail

SRC="${SRC:-/autodl-pub/data/nuScenes/Fulldatasetv1.0/Mini/v1.0-mini.tgz}"
DST="${DST:-/root/autodl-tmp/data/nuscenes}"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: source archive not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"
echo "Extracting $SRC -> $DST ..."
tar -xzf "$SRC" -C "$DST"

echo "Done. Expected layout:"
echo "  $DST/v1.0-mini/   (metadata json)"
echo "  $DST/samples/     (keyframe images)"
echo "  $DST/sweeps/      (intermediate frames)"
ls -1 "$DST" || true
