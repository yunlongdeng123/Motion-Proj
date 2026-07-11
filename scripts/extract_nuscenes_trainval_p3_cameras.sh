#!/usr/bin/env bash
# 从共享归档增量提取 P3 五个非前视相机关键帧，并输出逐相机完整性 manifest。
set -euo pipefail

SRC="${SRC:-/autodl-pub/data/nuScenes/Fulldatasetv1.0/Trainval}"
DST="${DST:-/root/autodl-tmp/data/nuscenes}"
STATE_DIR="${STATE_DIR:-$DST/.trainval-p3-cameras-extract}"
MIN_FREE_GB="${MIN_FREE_GB:-55}"
CAMERAS=(CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK_LEFT CAM_BACK CAM_BACK_RIGHT)

mkdir -p "$DST" "$STATE_DIR"
exec 9>"$STATE_DIR/extract.lock"
flock -n 9 || { echo "ERROR: 已有 P3 相机提取任务运行中" >&2; exit 1; }

free_gb="$(df --output=avail -BG "$DST" | awk 'NR==2 {gsub(/G/, "", $1); print $1}')"
(( free_gb >= MIN_FREE_GB )) || { echo "ERROR: 可用空间 ${free_gb}GB 不足" >&2; exit 1; }

for shard in $(seq -w 1 10); do
  archive="$SRC/v1.0-trainval${shard}_blobs.tgz"
  marker="$STATE_DIR/shard-${shard}.complete"
  [[ -f "$marker" ]] && continue
  [[ -f "$archive" ]] || { echo "ERROR: 归档不存在: $archive" >&2; exit 1; }
  patterns=()
  for camera in "${CAMERAS[@]}"; do patterns+=("samples/$camera/*"); done
  tar -xzf "$archive" -C "$DST" --wildcards --skip-old-files "${patterns[@]}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"$marker"
done

python - "$DST" "$STATE_DIR/manifest.json" "${CAMERAS[@]}" <<'PY'
import json, os, sys
from pathlib import Path

root, output, *channels = sys.argv[1:]
root, output = Path(root), Path(output)
meta = root / "v1.0-trainval"
sensors = {row["token"]: row["channel"] for row in json.load((meta / "sensor.json").open())}
calibrated = {row["token"]: sensors[row["sensor_token"]]
              for row in json.load((meta / "calibrated_sensor.json").open())}
expected = {channel: [] for channel in channels}
for row in json.load((meta / "sample_data.json").open()):
    channel = calibrated[row["calibrated_sensor_token"]]
    if row["is_key_frame"] and channel in expected:
        expected[channel].append(row["filename"])
rows = {}
for channel, files in expected.items():
    missing = [name for name in files if not (root / name).is_file()]
    rows[channel] = {"expected": len(files), "present": len(files) - len(missing),
                     "missing": len(missing), "complete": not missing,
                     "bytes": sum(os.path.getsize(root / name) for name in files if (root / name).is_file())}
manifest = {"schema_version": 1, "dataroot": str(root), "cameras": rows,
            "complete": all(row["complete"] for row in rows.values())}
output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
if not manifest["complete"]:
    raise SystemExit("ERROR: P3 相机提取不完整")
PY
