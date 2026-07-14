#!/usr/bin/env bash
# 克隆 E0 等诊断所需的第三方仓库到固定路径。
set -euo pipefail

ROOT="${AUTODL_TMP:-/root/autodl-tmp}"
THIRD_PARTY="${ROOT}/third_party"
COTRACKER_DIR="${THIRD_PARTY}/co-tracker"
COTRACKER_COMMIT="82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
COTRACKER_REPO="https://github.com/facebookresearch/co-tracker.git"

mkdir -p "${THIRD_PARTY}"

if [[ -d "${COTRACKER_DIR}/.git" ]]; then
  echo "[co-tracker] 已存在，checkout ${COTRACKER_COMMIT}"
  git -C "${COTRACKER_DIR}" fetch --depth 1 origin "${COTRACKER_COMMIT}" 2>/dev/null || git -C "${COTRACKER_DIR}" fetch origin
  git -C "${COTRACKER_DIR}" checkout "${COTRACKER_COMMIT}"
else
  echo "[co-tracker] 克隆 ${COTRACKER_REPO} @ ${COTRACKER_COMMIT}"
  git clone "${COTRACKER_REPO}" "${COTRACKER_DIR}"
  git -C "${COTRACKER_DIR}" checkout "${COTRACKER_COMMIT}"
fi

echo "[co-tracker] 完成: $(git -C "${COTRACKER_DIR}" rev-parse --short HEAD)"
echo "E0 权重需单独下载到 ${COTRACKER_DIR}/checkpoints/，见 docs/THIRD_PARTY.md"
