#!/usr/bin/env bash
# 采集 AD-GS 环境产物，对应 DYNAMIC_DRIVING_RECONSTRUCTION_PLAN_V1.md 4.5 节清单。
# 用法: bash scripts/env_report_adgs.sh [输出目录]
set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/root/autodl-tmp/envs/adgs}
ADGS_ROOT=${ADGS_ROOT:-/root/autodl-tmp/third_party/AD-GS}
PYTORCH3D_ROOT=${PYTORCH3D_ROOT:-/root/autodl-tmp/third_party/pytorch3d-v0.7.2}
MICROMAMBA=${MICROMAMBA:-/root/autodl-tmp/bin/micromamba}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/root/autodl-tmp/micromamba}

OUT=${1:-/root/autodl-tmp/runs/dynamic_recon/DR-M2-ENV-ASSET-01/$(date +%Y%m%dT%H%M%S)/environment}
mkdir -p "$OUT"

echo "[1/8] 主机与驱动"
nvidia-smi                                  > "$OUT/gpu_driver.txt" 2>&1
"$CUDA_HOME/bin/nvcc" --version             > "$OUT/nvcc_version.txt" 2>&1
gcc --version                               > "$OUT/gcc_version.txt" 2>&1
{ echo "nproc: $(nproc)"; free -g; df -h /root/autodl-tmp; } > "$OUT/host_resources.txt" 2>&1
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.events \
                                            > "$OUT/cgroup.txt" 2>&1

echo "[2/8] 仓库来源与 commit"
{
  for repo in "$ADGS_ROOT" "$PYTORCH3D_ROOT"; do
    echo "### $repo"
    echo "url:    $(git -C "$repo" config --get remote.origin.url)"
    echo "commit: $(git -C "$repo" rev-parse HEAD)"
    echo "describe: $(git -C "$repo" describe --tags --always 2>/dev/null || echo n/a)"
    echo
  done
} > "$OUT/repo_url_commit.txt"

# 排除运行时重新生成的 .pyc（upstream 误将 __pycache__ 提交进仓库），只保留真实源码改动
git -C "$ADGS_ROOT" diff -- . ':(exclude)*__pycache__*' > "$OUT/git_diff.txt" 2>&1 || true
git -C "$ADGS_ROOT" status --short                      > "$OUT/git_status.txt" 2>&1 || true

{
  echo "### AD-GS 声明的 git submodule"
  git -C "$ADGS_ROOT" submodule status 2>&1
  echo "(空表示 AD-GS 未声明任何 submodule)"
  echo
  echo "### submodules/ 下的组件（upstream 直接提交在仓库内，非 submodule）"
  echo "depth-diff-gaussian-rasterization: vendored in AD-GS commit $(git -C "$ADGS_ROOT" rev-parse HEAD)"
  echo "simple-knn:                        vendored in AD-GS commit $(git -C "$ADGS_ROOT" rev-parse HEAD)"
  echo "third_party/glm:                   .gitmodules 声明 url=https://github.com/g-truc/glm.git，"
  echo "                                   但实际由 AD-GS 直接 tracked $(git -C "$ADGS_ROOT" ls-files submodules/depth-diff-gaussian-rasterization/third_party/glm | wc -l) 个文件"
} > "$OUT/submodule_commits.txt"

echo "[3/8] conda 层显式包列表"
"$MICROMAMBA" env export -p "$ENV_PREFIX" --explicit --md5 > "$OUT/conda_explicit.txt" 2>&1
"$MICROMAMBA" env export -p "$ENV_PREFIX"                  > "$OUT/conda_env_export.yaml" 2>&1
"$MICROMAMBA" list -p "$ENV_PREFIX"                        > "$OUT/conda_list.txt" 2>&1

echo "[4/8] pip freeze"
"$ENV_PREFIX/bin/pip" freeze                > "$OUT/pip_freeze.txt" 2>&1

echo "[5/8] python 版本"
"$ENV_PREFIX/bin/python" -V                 > "$OUT/python_version.txt" 2>&1

echo "[6/8] torch / cuda 版本矩阵"
"$ENV_PREFIX/bin/python" - > "$OUT/torch_cuda_versions.txt" 2>&1 <<'PY'
import torch, torchvision, torchaudio
print("torch            ", torch.__version__)
print("torchvision      ", torchvision.__version__)
print("torchaudio       ", torchaudio.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cudnn            ", torch.backends.cudnn.version())
print("arch_list        ", torch.cuda.get_arch_list())
print("device           ", torch.cuda.get_device_name(0))
print("capability       ", torch.cuda.get_device_capability(0))
import pytorch3d, diff_gaussian_rasterization, simple_knn
print("pytorch3d        ", pytorch3d.__version__)
print("rasterizer       ", diff_gaussian_rasterization.__file__)
print("simple_knn       ", simple_knn.__file__)
PY

echo "[7/8] 编译产物 so 清单"
{
  find "$ENV_PREFIX/lib/python3.7/site-packages" -maxdepth 2 -name "*.so" \
    \( -path "*pytorch3d*" -o -path "*diff_gaussian*" -o -path "*simple_knn*" \) 2>/dev/null
  find "$ADGS_ROOT/submodules" -name "*.so" 2>/dev/null
} | while read -r f; do echo "$(sha256sum "$f")"; done > "$OUT/cuda_ext_sha256.txt" 2>&1 || true

echo "[8/8] smoke 测试"
cd "$ADGS_ROOT"
OMP_NUM_THREADS=16 TORCH_CUDA_ARCH_LIST="8.6+PTX" CUDA_HOME="$CUDA_HOME" \
  "$ENV_PREFIX/bin/python" /root/autodl-tmp/motion_proj/scripts/smoke_adgs_env.py \
  > "$OUT/smoke.log" 2>&1 && SMOKE_RC=0 || SMOKE_RC=$?

echo
echo "产物目录: $OUT"
echo "smoke 退出码: $SMOKE_RC"
tail -3 "$OUT/smoke.log"
ls -la "$OUT"
exit "$SMOKE_RC"
