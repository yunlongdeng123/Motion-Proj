#!/usr/bin/env bash
set -Eeuo pipefail

RUN_DIR=${1:?usage: run_dr_v2_m3_nvdiffrast_rebuild.sh RUN_DIR}
PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/motion_proj}
SOURCE_ROOT=/root/autodl-tmp/third_party/nvdiffrast
ENV_ROOT=/root/autodl-tmp/envs/drivestudio
PYTHON=/root/autodl-tmp/envs/motionproj/bin/python
BUILD_PYTHON="$ENV_ROOT/bin/python"
TASK_ID=$(basename "$(dirname "$RUN_DIR")")
INSTANCE_ID=$(basename "$RUN_DIR")
INSTALLED_BINARY="$ENV_ROOT/lib/python3.9/site-packages/_nvdiffrast_c.cpython-39-x86_64-linux-gnu.so"
BACKUP_ROOT="/root/autodl-tmp/backups/drivestudio_m3/$INSTANCE_ID/nvdiffrast_pre_sm86_rebuild"

if [[ "$TASK_ID" != "DR-V2-M3-EDIT-BASELINE-01" ]]; then
  echo "unexpected task parent: $TASK_ID" >&2
  exit 64
fi
if [[ -e "$RUN_DIR/manifest.json" || -e "$RUN_DIR/terminal.json" ]]; then
  echo "refuse to overwrite initialized run: $RUN_DIR" >&2
  exit 73
fi

mkdir -p "$RUN_DIR"/{environment,logs,source_snapshot,stages} "$BACKUP_ROOT"
cp "$PROJECT_ROOT/scripts/run_dr_v2_m3_nvdiffrast_rebuild.sh" "$RUN_DIR/source_snapshot/"
cp "$PROJECT_ROOT/scripts/finalize_dr_v2_m3_run.py" "$RUN_DIR/source_snapshot/"
cp "$SOURCE_ROOT/setup.py" "$RUN_DIR/source_snapshot/nvdiffrast_setup.py"

"$PYTHON" - "$RUN_DIR" "$TASK_ID" "$INSTANCE_ID" "$SOURCE_ROOT" <<'PY'
import datetime
import json
import subprocess
import sys
from pathlib import Path

run_dir, task_id, instance_id, source_root = map(Path, sys.argv[1:])
now = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
commit = subprocess.check_output(
    ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
).strip()
status = subprocess.check_output(
    ["git", "-C", str(source_root), "status", "--short"], text=True
).splitlines()
(run_dir / "manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "task_id": str(task_id),
            "component": "DriveStudio nvdiffrast SM 8.6 environment repair",
            "instance_id": str(instance_id),
            "status": "running",
            "started_at": now,
            "source_commit": commit,
            "source_git_status": status,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
(run_dir / "terminal.json").write_text(
    json.dumps({"status": "running", "updated_at": now, "failure": None}, indent=2)
    + "\n",
    encoding="utf-8",
)
(run_dir / "resolved.yaml").write_text(
    "\n".join(
        [
            f"task_id: {task_id}",
            f"instance_id: {instance_id}",
            "component: DriveStudio nvdiffrast SM 8.6 environment repair",
            f"source_root: {source_root}",
            f"source_commit: {commit}",
            "torch_cuda_arch_list: 8.6+PTX",
            "max_jobs: 4",
            "semantic_patch: false",
            "",
        ]
    ),
    encoding="utf-8",
)
PY

if [[ ! -s "$INSTALLED_BINARY" ]]; then
  echo "installed nvdiffrast binary missing: $INSTALLED_BINARY" >&2
  exit 66
fi
cp -a "$INSTALLED_BINARY" "$BACKUP_ROOT/"
sha256sum "$INSTALLED_BINARY" "$BACKUP_ROOT/$(basename "$INSTALLED_BINARY")" \
  > "$RUN_DIR/environment/pre_rebuild_binaries.sha256"
git -C "$SOURCE_ROOT" rev-parse HEAD > "$RUN_DIR/environment/source_commit.txt"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader \
  > "$RUN_DIR/environment/gpu.txt"
"$BUILD_PYTHON" - <<'PY' > "$RUN_DIR/environment/torch.txt"
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY

export PATH="$ENV_ROOT/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6+PTX
export MAX_JOBS=4
export OMP_NUM_THREADS=8
set +e
(
  cd "$SOURCE_ROOT"
  "$BUILD_PYTHON" setup.py build_ext --inplace --force
) > "$RUN_DIR/logs/build.log" 2>&1
build_rc=$?
set -e

verify_rc=125
BUILT_BINARY=$(find "$SOURCE_ROOT" -maxdepth 1 -type f -name '_nvdiffrast_c*.so' -print -quit)
if [[ "$build_rc" -eq 0 && -n "$BUILT_BINARY" && -s "$BUILT_BINARY" ]]; then
  cp -a "$BUILT_BINARY" "$INSTALLED_BINARY"
  set +e
  CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/root/autodl-tmp/third_party/drivestudio \
    "$BUILD_PYTHON" - <<'PY' > "$RUN_DIR/logs/forward_backward.log" 2>&1
import json
import torch
from models.modules import EnvLight

module = EnvLight(
    class_name="Sky", resolution=8, device=torch.device("cuda")
).to("cuda")
viewdirs = torch.randn(6, 7, 3, device="cuda", requires_grad=True)
output = module({"viewdirs": viewdirs})
loss = output.square().mean()
loss.backward()
torch.cuda.synchronize()
payload = {
    "status": "PASS",
    "output_shape": list(output.shape),
    "output_finite": bool(torch.isfinite(output).all()),
    "viewdirs_grad_finite": bool(torch.isfinite(viewdirs.grad).all()),
    "base_grad_finite": bool(torch.isfinite(module.base.grad).all()),
    "loss": float(loss.detach()),
}
assert payload["output_finite"]
assert payload["viewdirs_grad_finite"]
assert payload["base_grad_finite"]
print(json.dumps(payload, sort_keys=True))
PY
  verify_rc=$?
  set -e
fi

if [[ -s "$INSTALLED_BINARY" ]]; then
  sha256sum "$INSTALLED_BINARY" > "$RUN_DIR/environment/post_rebuild_binary.sha256"
fi

"$PYTHON" - "$RUN_DIR/stages/nvdiffrast_rebuild.json" "$build_rc" "$verify_rc" \
  "$INSTALLED_BINARY" "$BUILT_BINARY" "$BACKUP_ROOT" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

output, build_rc, verify_rc, installed, built, backup = sys.argv[1:]
installed_path = Path(installed)

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()

done = int(build_rc) == 0 and int(verify_rc) == 0 and installed_path.is_file()
payload = {
    "stage": "nvdiffrast_rebuild",
    "status": "done" if done else "blocked",
    "build_return_code": int(build_rc),
    "verify_return_code": int(verify_rc),
    "source_commit": "253ac4fcea7de5f396371124af597e6cc957bfae",
    "torch_cuda_arch_list": "8.6+PTX",
    "semantic_patch": False,
    "installed_binary": str(installed_path),
    "built_binary": built or None,
    "backup_root": backup,
    "installed_sha256": digest(installed_path) if installed_path.is_file() else None,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
}
temporary = Path(output).with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
PY

if [[ "$build_rc" -eq 0 && "$verify_rc" -eq 0 ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/finalize_dr_v2_m3_run.py" \
    --run-dir "$RUN_DIR" \
    --task-id "$TASK_ID" \
    --status done \
    --summary "nvdiffrast was rebuilt from the frozen upstream source for SM 8.6; the DriveStudio EnvLight texture forward/backward smoke passed. No renderer semantics were patched." \
    --required-stage nvdiffrast_rebuild
  exit 0
fi

"$PYTHON" "$PROJECT_ROOT/scripts/finalize_dr_v2_m3_run.py" \
  --run-dir "$RUN_DIR" \
  --task-id "$TASK_ID" \
  --status blocked \
  --summary "nvdiffrast SM 8.6 rebuild or the EnvLight CUDA forward/backward verification failed; no new native training run was started." \
  --failure-code M3_NVDIFFRAST_SM86_REBUILD_FAILED \
  --failure-detail "build_rc=$build_rc verify_rc=$verify_rc" \
  --evidence logs/build.log \
  --evidence logs/forward_backward.log
exit 2
