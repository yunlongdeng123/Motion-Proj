#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT=/root/autodl-tmp/motion_proj
DGGT_ROOT=/root/autodl-tmp/third_party/dggt
ENV_PREFIX=/root/autodl-tmp/envs/dggt-v2
CHECKPOINT_PRELOAD=/root/autodl-tmp/checkpoints/dggt_preload/model_latest_nuscenes.pt
CHECKPOINT_ROOT=/root/autodl-tmp/checkpoints/dggt-v2
CHECKPOINT_LINK="$CHECKPOINT_ROOT/model_latest_nuscenes.pt"
CONSTRAINTS="$PROJECT_ROOT/configs/env/dggt_v2_constraints.txt"
DGGT_COMMIT=a3276d2bbe4cbb03bcc117830b1836110a27adeb
MODEL_REVISION=735ac9a6486057b1eb886c33a8c6dc79e0b43214
CHECKPOINT_BYTES=5411266466
CHECKPOINT_SHA256=fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9
CURRENT_STAGE=initialize

usage() {
  printf '%s\n' "用法: $0 --run-dir /root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M1-DGGT-REPAIR-01/<instance>"
}

if [[ $# -ne 2 || "$1" != "--run-dir" ]]; then
  usage >&2
  exit 2
fi
RUN_DIR="$2"
if [[ "$RUN_DIR" != /root/autodl-tmp/runs/dynamic_editing_v2/DR-V2-M1-DGGT-REPAIR-01/* ]]; then
  printf '%s\n' "run 路径不属于 M1: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  printf '%s\n' "run 已存在，禁止覆盖: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$ENV_PREFIX" ]]; then
  printf '%s\n' "dggt-v2 环境已存在，禁止就地续跑: $ENV_PREFIX" >&2
  exit 2
fi

mkdir -p \
  "$RUN_DIR/environment/bootstrap" "$RUN_DIR/logs" "$RUN_DIR/stages" \
  "$RUN_DIR/source_snapshot" "$RUN_DIR/artifacts"

write_terminal() {
  local status="$1"
  local failure="$2"
  /root/miniconda3/bin/python - "$RUN_DIR/terminal.json" "$status" "$failure" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "updated_at": dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat(),
    "failure": sys.argv[3] or None,
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY
}

on_error() {
  local rc=$?
  write_terminal blocked "stage=$CURRENT_STAGE rc=$rc"
  exit "$rc"
}
trap on_error ERR

resource_snapshot() {
  local name="$1"
  {
    date --iso-8601=seconds
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
    cat /sys/fs/cgroup/memory.max
    cat /sys/fs/cgroup/memory.current
    cat /sys/fs/cgroup/memory.events
    df -h /root/autodl-tmp
  } > "$RUN_DIR/environment/resource_${name}.txt"
}

run_stage() {
  local name="$1"
  shift
  CURRENT_STAGE="$name"
  local started
  started="$(date --iso-8601=seconds)"
  set +e
  "$@" > "$RUN_DIR/logs/${name}.log" 2>&1
  local rc=$?
  set -e
  /root/miniconda3/bin/python - "$RUN_DIR/stages/${name}.json" "$name" "$started" "$rc" "$@" <<'PY'
import datetime as dt
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "stage": sys.argv[2],
    "started_at": sys.argv[3],
    "finished_at": dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat(),
    "return_code": int(sys.argv[4]),
    "status": "done" if int(sys.argv[4]) == 0 else "blocked",
    "command": sys.argv[5:],
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY
  if [[ $rc -ne 0 ]]; then
    return "$rc"
  fi
}

cd "$PROJECT_ROOT"
write_terminal running ""
resource_snapshot before
git status --short --branch > "$RUN_DIR/source_snapshot/project_git_status.txt"
git rev-parse HEAD > "$RUN_DIR/source_snapshot/project_commit.txt"
git -C "$DGGT_ROOT" status --short --branch > "$RUN_DIR/source_snapshot/dggt_git_status.txt"
git -C "$DGGT_ROOT" remote -v > "$RUN_DIR/source_snapshot/dggt_remotes.txt"
test "$(git -C "$DGGT_ROOT" rev-parse HEAD)" = "$DGGT_COMMIT"
git -C "$DGGT_ROOT" diff --binary > "$RUN_DIR/source_snapshot/dggt_diff.patch"
cp "$DGGT_ROOT/LICENSE" "$RUN_DIR/source_snapshot/DGGT_LICENSE"
cp "$DGGT_ROOT/README.md" "$RUN_DIR/source_snapshot/DGGT_README.md"
cp "$CONSTRAINTS" "$RUN_DIR/source_snapshot/dggt_v2_constraints.txt"

CURRENT_STAGE=bootstrap
# shellcheck disable=SC1091
source "$PROJECT_ROOT/scripts/bootstrap_autodl_v2.sh" \
  --report-dir "$RUN_DIR/environment/bootstrap"
# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh

CURRENT_STAGE=model_provenance
curl -L --fail --retry 3 --connect-timeout 20 --max-time 120 \
  "https://hf-mirror.com/api/models/xiaomi-research/dggt/revision/$MODEL_REVISION" \
  -o "$RUN_DIR/source_snapshot/model_revision.json"
/root/miniconda3/bin/python - "$RUN_DIR/source_snapshot/model_revision.json" "$MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["sha"] == sys.argv[2]
assert payload["cardData"]["license"] == "cc-by-nc-4.0"
siblings = {row["rfilename"] for row in payload["siblings"]}
assert "model_latest_nuscenes.pt" in siblings
PY
test "$(stat -c %s "$CHECKPOINT_PRELOAD")" = "$CHECKPOINT_BYTES"
test "$(sha256sum "$CHECKPOINT_PRELOAD" | cut -d " " -f 1)" = "$CHECKPOINT_SHA256"
mkdir -p "$CHECKPOINT_ROOT"
if [[ -e "$CHECKPOINT_LINK" ]]; then
  test "$(stat -c %s "$CHECKPOINT_LINK")" = "$CHECKPOINT_BYTES"
  test "$(sha256sum "$CHECKPOINT_LINK" | cut -d " " -f 1)" = "$CHECKPOINT_SHA256"
  test "$(stat -c %i "$CHECKPOINT_PRELOAD")" = "$(stat -c %i "$CHECKPOINT_LINK")"
else
  ln "$CHECKPOINT_PRELOAD" "$CHECKPOINT_LINK"
fi
test "$(stat -c %i "$CHECKPOINT_PRELOAD")" = "$(stat -c %i "$CHECKPOINT_LINK")"

run_stage env_create conda create -y -p "$ENV_PREFIX" python=3.10 pip

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=3
run_stage env_torch "$ENV_PREFIX/bin/python" -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
run_stage env_requirements "$ENV_PREFIX/bin/python" -m pip install \
  --constraint "$CONSTRAINTS" -r "$DGGT_ROOT/requirements.txt" ninja

CUDA_CHANNEL=https://conda.anaconda.org/nvidia/label/cuda-12.1.0
run_stage env_cuda_toolkit conda install -y -p "$ENV_PREFIX" \
  --override-channels -c "$CUDA_CHANNEL" \
  cuda-nvcc=12.1.66 cuda-cudart-dev=12.1.55 cuda-cccl=12.1.55

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export OMP_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST=8.6
NVIDIA_SITE="$ENV_PREFIX/lib/python3.10/site-packages/nvidia"
NVIDIA_INCLUDE_PATHS="$(
  find "$NVIDIA_SITE" -mindepth 2 -maxdepth 2 -type d -name include -print \
    | sort | paste -sd: -
)"
NVIDIA_LIBRARY_PATHS="$(
  find "$NVIDIA_SITE" -mindepth 2 -maxdepth 2 -type d -name lib -print \
    | sort | paste -sd: -
)"
export CPATH="$NVIDIA_INCLUDE_PATHS${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NVIDIA_LIBRARY_PATHS${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$NVIDIA_LIBRARY_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run_stage cuda_toolchain_smoke bash -c \
  "test -x '$CUDA_HOME/bin/nvcc' && test -f '$NVIDIA_SITE/cusparse/include/cusparse.h' && '$CUDA_HOME/bin/nvcc' --version | grep -F 'release 12.1'"
run_stage torch_cuda_smoke "$ENV_PREFIX/bin/python" -c \
  "import torch; assert torch.__version__.startswith('2.4.1'); assert torch.version.cuda == '12.1'; assert torch.cuda.is_available(); x=torch.randn(1024,1024,device='cuda',requires_grad=True); y=(x@x).mean(); y.backward(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),float(y),float(x.grad.abs().mean()))"
run_stage env_pointops2_upstream bash -c \
  "cd '$DGGT_ROOT/third_party/pointops2' && '$ENV_PREFIX/bin/python' setup.py install"

run_stage pointops2_cuda_smoke "$ENV_PREFIX/bin/python" -c \
  "import torch; from pointops2.pointops import grouping; x=torch.randn(4,3,device='cuda',requires_grad=True).contiguous(); idx=torch.tensor([[0,1],[2,3]],device='cuda',dtype=torch.int32).contiguous(); y=grouping(x,idx); loss=y.square().sum(); loss.backward(); assert x.grad is not None and torch.isfinite(x.grad).all(); print(tuple(y.shape),float(loss),float(x.grad.abs().sum()))"

CURRENT_STAGE=environment_audit
"$ENV_PREFIX/bin/python" -V > "$RUN_DIR/environment/python_version.txt" 2>&1
"$ENV_PREFIX/bin/python" -m pip freeze > "$RUN_DIR/environment/pip_freeze.txt"
conda list -p "$ENV_PREFIX" --explicit > "$RUN_DIR/environment/conda_explicit.txt"
"$ENV_PREFIX/bin/python" -c \
  "import torch,torchvision,torchaudio; print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('torchaudio',torchaudio.__version__); print('torch_cuda',torch.version.cuda); print('cuda_available',torch.cuda.is_available())" \
  > "$RUN_DIR/environment/torch_cuda_versions.txt"
nvcc --version > "$RUN_DIR/environment/nvcc_version.txt"
{
  printf 'CUDA_HOME=%s\n' "$CUDA_HOME"
  printf 'TORCH_CUDA_ARCH_LIST=%s\n' "$TORCH_CUDA_ARCH_LIST"
  printf 'CPATH=%s\n' "$CPATH"
  printf 'LIBRARY_PATH=%s\n' "$LIBRARY_PATH"
  printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
} > "$RUN_DIR/environment/cuda_build_paths.txt"
gcc --version > "$RUN_DIR/environment/gcc_version.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  > "$RUN_DIR/environment/gpu_driver.txt"
sha256sum "$CHECKPOINT_LINK" > "$RUN_DIR/environment/checkpoint_sha256.txt"
stat -c '%n %s %i %h' "$CHECKPOINT_PRELOAD" "$CHECKPOINT_LINK" \
  > "$RUN_DIR/environment/checkpoint_hardlink.txt"
resource_snapshot after

/root/miniconda3/bin/python - "$RUN_DIR/manifest.json" "$RUN_DIR" <<'PY'
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
payload = {
    "schema_version": 1,
    "task_id": "DR-V2-M1-DGGT-REPAIR-01",
    "instance_id": run_dir.name,
    "created_at": dt.datetime.now(
        dt.timezone(dt.timedelta(hours=8))
    ).isoformat(),
    "project_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/root/autodl-tmp/motion_proj", text=True
    ).strip(),
    "dggt_commit": "a3276d2bbe4cbb03bcc117830b1836110a27adeb",
    "model_revision": "735ac9a6486057b1eb886c33a8c6dc79e0b43214",
    "checkpoint_sha256": "fd15644b3a878849470cbf5f0f9eae39167cfec1b853092898ae754c4f3acde9",
    "cuda_toolkit_channel": "https://conda.anaconda.org/nvidia/label/cuda-12.1.0",
    "cuda_toolkit": "12.1",
    "seed": 0,
    "run_dir": str(run_dir),
    "status": "running",
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
os.replace(temporary, path)
PY

printf '%s\n' "DGGT V2 环境与 pointops2 upstream smoke 完成: $RUN_DIR"
