#!/usr/bin/env bash
# 从 AutoDL 共享归档中仅提取前视主实验需要的关键帧与 LiDAR。
set -euo pipefail

SRC="${SRC:-/autodl-pub/data/nuScenes/Fulldatasetv1.0/Trainval}"
DST="${DST:-/root/autodl-tmp/data/nuscenes}"
STATE_DIR="${STATE_DIR:-$DST/.trainval-front-extract}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

mkdir -p "$DST" "$STATE_DIR"
exec 9>"$STATE_DIR/extract.lock"
if ! flock -n 9; then
  echo "ERROR: 已有 trainval 前视提取任务运行中" >&2
  exit 1
fi

meta_archive="$SRC/v1.0-trainval_meta.tgz"
if [[ ! -f "$meta_archive" ]]; then
  echo "ERROR: metadata 归档不存在: $meta_archive" >&2
  exit 1
fi

free_gb="$(df --output=avail -BG "$DST" | awk 'NR==2 {gsub(/G/, "", $1); print $1}')"
if (( free_gb < MIN_FREE_GB )); then
  echo "ERROR: 可用空间 ${free_gb}GB，小于安全下限 ${MIN_FREE_GB}GB" >&2
  exit 1
fi

if [[ ! -f "$DST/v1.0-trainval/scene.json" ]]; then
  echo "提取 trainval metadata"
  tar -xzf "$meta_archive" -C "$DST"
fi

for shard in $(seq -w 1 10); do
  archive="$SRC/v1.0-trainval${shard}_blobs.tgz"
  marker="$STATE_DIR/shard-${shard}.complete"
  if [[ -f "$marker" ]]; then
    echo "跳过已完成 shard ${shard}"
    continue
  fi
  if [[ ! -f "$archive" ]]; then
    echo "ERROR: blob 归档不存在: $archive" >&2
    exit 1
  fi

  echo "提取 shard ${shard}: CAM_FRONT + LIDAR_TOP"
  tar -xzf "$archive" -C "$DST" \
    --wildcards --skip-old-files \
    "samples/CAM_FRONT/*" \
    "samples/LIDAR_TOP/*"
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"$marker"
done

python - "$DST" "$STATE_DIR/summary.json" <<'PY'
import json
import os
import sys
from pathlib import Path

dataroot = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
metadata = dataroot / "v1.0-trainval"

with (metadata / "sensor.json").open(encoding="utf-8") as handle:
    sensors = {row["token"]: row["channel"] for row in json.load(handle)}
with (metadata / "calibrated_sensor.json").open(encoding="utf-8") as handle:
    calibrated = {
        row["token"]: sensors[row["sensor_token"]]
        for row in json.load(handle)
    }
with (metadata / "sample_data.json").open(encoding="utf-8") as handle:
    sample_data = json.load(handle)

channels = {"CAM_FRONT", "LIDAR_TOP"}
expected = {channel: [] for channel in channels}
for row in sample_data:
    channel = calibrated[row["calibrated_sensor_token"]]
    if row["is_key_frame"] and channel in channels:
        expected[channel].append(row["filename"])

missing = {
    channel: [name for name in names if not (dataroot / name).is_file()]
    for channel, names in expected.items()
}
summary = {
    "dataroot": str(dataroot),
    "counts": {channel: len(names) for channel, names in sorted(expected.items())},
    "missing": {channel: len(names) for channel, names in sorted(missing.items())},
    "bytes": {
        channel: sum(os.path.getsize(dataroot / name) for name in names if (dataroot / name).is_file())
        for channel, names in sorted(expected.items())
    },
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
if any(missing.values()):
    raise SystemExit("ERROR: 精选数据提取不完整")
PY

echo "trainval 前视数据提取与完整性检查完成"
