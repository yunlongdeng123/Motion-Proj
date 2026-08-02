#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
RUN_DIR=${1:?usage: run_dr_v2_m3_prepare.sh RUN_DIR}
INSTANCE_ID=$(basename "$RUN_DIR")
TASK_ID=$(basename "$(dirname "$RUN_DIR")")
RAW_ROOT=/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_raw_scene0230
MANIFEST_DIR=/root/autodl-tmp/data/dynamic_editing_v2/manifests

mkdir -p "$RUN_DIR"/{artifacts,environment,logs,qa,source_snapshot,stages}
for source in \
  scripts/audit_dr_v2_m3_baseline.py \
  scripts/build_dr_v2_drivestudio_registry.py \
  scripts/finalize_dr_v2_m3_run.py \
  scripts/prepare_dr_v2_drivestudio_scene.py \
  scripts/probe_dr_v2_drivestudio_actor.py \
  scripts/record_dr_v2_m3_asset_reuse.py \
  scripts/record_dr_v2_m3_checkpoint_recovery.py \
  scripts/run_dr_v2_drivestudio_edit_smoke.py \
  scripts/run_dr_v2_m3_native_smoke.sh \
  scripts/run_dr_v2_m3_native_attempt.sh \
  scripts/run_dr_v2_m3_nvdiffrast_rebuild.sh \
  scripts/run_dr_v2_m3_post_training.py \
  scripts/run_dr_v2_m3_prepare.sh \
  scripts/run_dr_v2_m3_recovery.sh \
  scripts/run_dr_v2_m3_training.py
do
  destination="$RUN_DIR/source_snapshot/$source"
  mkdir -p "$(dirname "$destination")"
  cp "$PROJECT_ROOT/$source" "$destination"
done

"$PYTHON" - "$RUN_DIR/manifest.json" "$TASK_ID" "$INSTANCE_ID" "$PROJECT_ROOT" <<'PY'
import datetime
import json
import subprocess
import sys
from pathlib import Path

path, task_id, instance_id, project_root = sys.argv[1:]
payload = {
    "schema_version": 1,
    "task_id": task_id,
    "component": "DriveStudio/StreetGS actor-aware native baseline",
    "instance_id": instance_id,
    "status": "running",
    "started_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "project_commit": subprocess.check_output(
        ["git", "-C", project_root, "rev-parse", "HEAD"], text=True
    ).strip(),
    "project_git_status": subprocess.check_output(
        ["git", "-C", project_root, "status", "--short"], text=True
    ).splitlines(),
    "path": "2_no_checkpoint_official_training",
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

"$PYTHON" - "$RUN_DIR/terminal.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "status": "running",
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "failure": None,
}, indent=2) + "\n")
PY

"$PYTHON" - "$RUN_DIR/resolved.yaml" "$TASK_ID" "$INSTANCE_ID" "$RAW_ROOT" <<'PY'
import sys
from pathlib import Path

path, task_id, instance_id, raw_root = sys.argv[1:]
Path(path).write_text(
    "\n".join(
        [
            f"task_id: {task_id}",
            f"instance_id: {instance_id}",
            "baseline: DriveStudio/StreetGS actor-aware native baseline",
            "scene_name: scene-0230",
            "scene_index: 179",
            "path: 2_no_checkpoint_official_training",
            f"raw_root: {raw_root}",
            "processed_root: /root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval",
            "upstream_root: /root/autodl-tmp/third_party/drivestudio",
            "upstream_commit: e59bda4fa681f829dbb1d65f0de582b0f633c450",
            "dataset_config: nuscenes/3cams",
            "method_config: configs/streetgs.yaml",
            "seed: 0",
            "extract_workers: 1",
            "",
        ]
    ),
    encoding="utf-8",
)
PY

resource_sample() {
  "$PYTHON" - "$RUN_DIR/resource.jsonl" <<'PY'
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader,nounits"],
    capture_output=True, text=True,
)
gpu_fields = [field.strip() for field in gpu.stdout.strip().split(",")] if gpu.returncode == 0 else []
def read_int(path):
    raw = Path(path).read_text().strip()
    return None if raw == "max" else int(raw)
disk = shutil.disk_usage("/root/autodl-tmp")
payload = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "memory_current_bytes": read_int("/sys/fs/cgroup/memory.current"),
    "memory_max_bytes": read_int("/sys/fs/cgroup/memory.max"),
    "memory_events": Path("/sys/fs/cgroup/memory.events").read_text().splitlines(),
    "disk_free_bytes": disk.free,
    "gpu": {
        "name": gpu_fields[0] if len(gpu_fields) > 0 else None,
        "driver": gpu_fields[1] if len(gpu_fields) > 1 else None,
        "memory_total_mib": int(gpu_fields[2]) if len(gpu_fields) > 2 else None,
        "memory_used_mib": int(gpu_fields[3]) if len(gpu_fields) > 3 else None,
    },
}
with Path(sys.argv[1]).open("a") as handle:
    handle.write(json.dumps(payload, sort_keys=True) + "\n")
PY
}

resource_monitor() {
  while true; do
    resource_sample
    sleep 10
  done
}

resource_sample
resource_monitor &
MONITOR_PID=$!
cleanup() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  resource_sample || true
}
trap cleanup EXIT

set +e
/root/autodl-tmp/envs/motionproj/bin/python \
  "$PROJECT_ROOT/scripts/prepare_dr_v2_drivestudio_scene.py" \
  --scene-name scene-0230 \
  --scene-index 179 \
  --out-root "$RAW_ROOT" \
  --manifest-dir "$MANIFEST_DIR" \
  --workers 1 \
  > "$RUN_DIR/logs/raw_prepare.log" 2>&1
rc=$?
set -e

"$PYTHON" - "$RUN_DIR/stages/raw_prepare.json" "$rc" "$MANIFEST_DIR/scene-0230_raw_manifest.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path
out, rc, manifest = sys.argv[1:]
payload = {
    "stage": "raw_prepare",
    "status": "done" if int(rc) == 0 else "failed",
    "return_code": int(rc),
    "manifest": manifest if Path(manifest).is_file() else None,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
PY
exit "$rc"
