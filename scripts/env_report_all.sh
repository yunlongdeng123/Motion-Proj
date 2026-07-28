#!/usr/bin/env bash
# 采集 M2 三个环境的产物并生成 summary.md
set -euo pipefail

RUN_DIR=${1:-/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/$(date +%Y%m%dT%H%M%S)}
ENV_DIR="$RUN_DIR/environment"
MICROMAMBA=${MICROMAMBA:-/root/autodl-tmp/bin/micromamba}
mkdir -p "$ENV_DIR"
SMOKE_FAILURES=0

echo "=== AD-GS 环境产物 ==="
if bash /root/autodl-tmp/motion_proj/scripts/env_report_adgs.sh "$ENV_DIR/adgs"; then
  echo 0 > "$ENV_DIR/adgs/smoke.rc"
else
  rc=$?
  echo "$rc" > "$ENV_DIR/adgs/smoke.rc"
  SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
fi

collect_simple() {
  local name=$1 prefix=$2 repo=$3 smoke_script=$4
  local out="$ENV_DIR/$name"
  mkdir -p "$out"
  echo "[collect] $name"
  "$prefix/bin/python" -V > "$out/python_version.txt" 2>&1
  "$prefix/bin/pip" freeze > "$out/pip_freeze.txt" 2>&1
  "$MICROMAMBA" env export -p "$prefix" --explicit --md5 > "$out/conda_explicit.txt" 2>&1
  "$MICROMAMBA" env export -p "$prefix" > "$out/conda_env_export.yaml" 2>&1
  {
    echo "url:    $(git -C "$repo" config --get remote.origin.url)"
    echo "commit: $(git -C "$repo" rev-parse HEAD)"
    echo "describe: $(git -C "$repo" describe --tags --always 2>/dev/null || echo n/a)"
  } > "$out/repo_url_commit.txt"
  git -C "$repo" diff -- . ':(exclude)*__pycache__*' > "$out/git_diff.txt" 2>&1 || true
  git -C "$repo" status --short > "$out/git_status.txt" 2>&1 || true
  git -C "$repo" submodule status > "$out/submodule_commits.txt" 2>&1 || true
  cp "$ENV_DIR/adgs/gpu_driver.txt" "$out/gpu_driver.txt"
  cp "$ENV_DIR/adgs/nvcc_version.txt" "$out/nvcc_version.txt"
  cp "$ENV_DIR/adgs/gcc_version.txt" "$out/gcc_version.txt"
  "$prefix/bin/python" - > "$out/torch_cuda_versions.txt" 2>&1 <<PY
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
  if [ -n "$smoke_script" ]; then
    OMP_NUM_THREADS=16 CUDA_HOME=/usr/local/cuda-11.8 \
      "$prefix/bin/python" "$smoke_script" > "$out/smoke.log" 2>&1 && rc=0 || rc=$?
    echo "$rc" > "$out/smoke.rc"
    if [ "$rc" -ne 0 ]; then
      SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
    fi
  fi
}

collect_simple adgs-dpt /root/autodl-tmp/envs/adgs-dpt \
  /root/autodl-tmp/third_party/Depth-Anything-V2 \
  /root/autodl-tmp/motion_proj/scripts/smoke_dpt_env.py

collect_simple adgs-sam /root/autodl-tmp/envs/adgs-sam \
  /root/autodl-tmp/third_party/Grounded-SAM-2 \
  ""

# SAM smoke 必须在仓库根目录运行
OMP_NUM_THREADS=16 CUDA_HOME=/usr/local/cuda-11.8 HF_ENDPOINT=https://hf-mirror.com \
  bash -c 'cd /root/autodl-tmp/third_party/Grounded-SAM-2 && \
  /root/autodl-tmp/envs/adgs-sam/bin/python /root/autodl-tmp/motion_proj/scripts/smoke_sam_env.py' \
  > "$ENV_DIR/adgs-sam/smoke.log" 2>&1 && rc=0 || rc=$?
echo "$rc" > "$ENV_DIR/adgs-sam/smoke.rc"
if [ "$rc" -ne 0 ]; then
  SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
fi

echo "[collect] pinned Grounding DINO Hugging Face model"
HF_HOME=/root/autodl-tmp/hf_cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=16 CUDA_HOME=/usr/local/cuda-11.8 \
  /root/autodl-tmp/envs/adgs-sam/bin/python \
  /root/autodl-tmp/motion_proj/scripts/smoke_grounding_dino_hf.py \
  > "$ENV_DIR/adgs-sam/grounding_dino_hf_smoke.log" 2>&1 && rc=0 || rc=$?
echo "$rc" > "$ENV_DIR/adgs-sam/grounding_dino_hf_smoke.rc"
if [ "$rc" -ne 0 ]; then
  SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
fi

echo "[collect] cotracker"
COTRACKER_OUT="$ENV_DIR/cotracker"
mkdir -p "$COTRACKER_OUT"
{
  echo "url:    $(git -C /root/autodl-tmp/third_party/co-tracker config --get remote.origin.url)"
  echo "commit: $(git -C /root/autodl-tmp/third_party/co-tracker rev-parse HEAD)"
} > "$COTRACKER_OUT/repo_url_commit.txt"
git -C /root/autodl-tmp/third_party/co-tracker diff -- . ':(exclude)*__pycache__*' \
  > "$COTRACKER_OUT/git_diff.txt" 2>&1 || true
git -C /root/autodl-tmp/third_party/co-tracker status --short \
  > "$COTRACKER_OUT/git_status.txt" 2>&1 || true
sha256sum /root/autodl-tmp/checkpoints/cotracker3/scaled_offline.pth \
  > "$COTRACKER_OUT/checkpoint_sha256.txt"
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 \
  /root/autodl-tmp/envs/adgs/bin/python \
  /root/autodl-tmp/motion_proj/scripts/smoke_cotracker_env.py \
  > "$COTRACKER_OUT/smoke.log" 2>&1 && rc=0 || rc=$?
echo "$rc" > "$COTRACKER_OUT/smoke.rc"
if [ "$rc" -ne 0 ]; then
  SMOKE_FAILURES=$((SMOKE_FAILURES + 1))
fi

{
  echo "# DR-M2-ENV-ASSET-01 环境安装 Summary"
  echo
  echo "生成时间: $(date -Iseconds)"
  echo "Run 目录: $RUN_DIR"
  echo
  echo "## 状态"
  echo
  echo "| 环境 | 路径 | Python | PyTorch | Smoke |"
  echo "|------|------|--------|---------|-------|"
  for e in adgs adgs-dpt adgs-sam cotracker; do
    py=$(head -1 "$ENV_DIR/$e/python_version.txt" 2>/dev/null || echo n/a)
    torch=$(grep "^torch" "$ENV_DIR/$e/torch_cuda_versions.txt" 2>/dev/null | head -1 || echo n/a)
    if grep -q "PASSED" "$ENV_DIR/$e/smoke.log" 2>/dev/null; then sm=PASS
    elif [ "$e" = "adgs" ] && grep -q "ALL SMOKE CHECKS PASSED" "$ENV_DIR/adgs/smoke.log" 2>/dev/null; then sm=PASS
    else sm=FAIL; fi
    if [ "$e" = "cotracker" ]; then
      env_path=/root/autodl-tmp/envs/adgs
      py=$(head -1 "$ENV_DIR/adgs/python_version.txt" 2>/dev/null || echo n/a)
      torch=$(grep "^torch" "$ENV_DIR/adgs/torch_cuda_versions.txt" 2>/dev/null | head -1 || echo n/a)
    else
      env_path=/root/autodl-tmp/envs/$e
    fi
    echo "| $e | $env_path | $py | $torch | $sm |"
  done
  echo
  echo "## 仓库 commit"
  echo
  for repo in AD-GS pytorch3d-v0.7.2 Depth-Anything-V2 Grounded-SAM-2 co-tracker; do
    echo "- **$repo**: \`$(git -C /root/autodl-tmp/third_party/$repo rev-parse HEAD)\`"
  done
  echo
  echo "## 权重 SHA-256"
  echo
  echo "- DPT Large: \`$(sha256sum /root/autodl-tmp/checkpoints/depth_anything_v2/depth_anything_v2_vitl.pth | awk '{print $1}')\`"
  echo "- SAM2.1 large: \`$(sha256sum /root/autodl-tmp/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt | awk '{print $1}')\`"
  echo "- GroundingDINO-T: \`$(sha256sum /root/autodl-tmp/third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth | awk '{print $1}')\`"
  echo "- CoTracker3 scaled offline: \`$(sha256sum /root/autodl-tmp/checkpoints/cotracker3/scaled_offline.pth | awk '{print $1}')\`"
  echo
  echo "## 兼容性补丁（偏离 upstream 的最小改动）"
  echo
  echo "1. **求解器**: 不用 \`conda-libmamba-solver\`（base conda 求解卡死），改用独立 micromamba 2.0.5"
  echo "2. **pytorch3d**: pin 到 v0.7.2（main 不支持 Python 3.7）"
  echo "3. **transformers (adgs-sam)**: pin 4.44.2（pip 默认装 5.x 与 torch 2.4.1 不兼容；upstream yaml 为 4.33.2）"
  echo "4. **pip cache**: 迁至 \`/root/autodl-tmp/pip_cache\`（\`/root/.cache/pip/wheels\` overlay 损坏）"
  echo
  echo "## 激活方式"
  echo
  echo '```bash'
  echo "source /root/miniconda3/etc/profile.d/conda.sh"
  echo "conda activate adgs        # AD-GS 主环境"
  echo "conda activate adgs-dpt    # Depth Anything V2"
  echo "conda activate adgs-sam    # Grounded-SAM-2"
  echo '```'
  echo
  echo "## 磁盘"
  echo
  df -h /root/autodl-tmp | tail -1
  du -sh /root/autodl-tmp/envs/adgs /root/autodl-tmp/envs/adgs-dpt /root/autodl-tmp/envs/adgs-sam
} > "$RUN_DIR/summary.md"

echo "完成: $RUN_DIR/summary.md"
cat "$RUN_DIR/summary.md"
if [ "$SMOKE_FAILURES" -ne 0 ]; then
  echo "smoke 失败数: $SMOKE_FAILURES" >&2
  exit 1
fi
