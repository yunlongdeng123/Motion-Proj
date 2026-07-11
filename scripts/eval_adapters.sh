#!/usr/bin/env bash
# 批量评估 LoRA checkpoint：用同一套固定 clip/seed 对比 base SVD 与各 adapter，
# 输出每个 (adapter, clip) 的视频 + 指标 JSON，以及跨 adapter 的 summary.json。
#
# 用法：
#   bash scripts/eval_adapters.sh
#   NUM_CLIPS=8 SEED=1234 bash scripts/eval_adapters.sh
#   ADAPTERS="base,adapter_step2000,adapter_final" bash scripts/eval_adapters.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/train/motionproj_v1.yaml}"
MODEL_PRETRAINED="${MODEL_PRETRAINED:-/root/autodl-tmp/weights/svd-xt}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/runs/motionproj_v1}"
CKPT_DIR="${CKPT_DIR:-$WORK_DIR/ckpts}"
OUT_DIR="${OUT_DIR:-$WORK_DIR/eval/adapters}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf_cache}"

SEED="${SEED:-1234}"
NUM_CLIPS="${NUM_CLIPS:-6}"
NUM_STEPS="${NUM_STEPS:-25}"
CLIP_INDICES="${CLIP_INDICES:-}"
# 默认对齐 base + 全部已保存 checkpoint（step500/1000/1500/2000/final）
ADAPTERS="${ADAPTERS:-base,adapter_step500,adapter_step1000,adapter_step1500,adapter_step2000,adapter_final}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_ENDPOINT HF_HOME

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate motionproj
fi

echo "== Motion-Proj adapter evaluation =="
echo "config:    $CONFIG"
echo "model:     $MODEL_PRETRAINED"
echo "ckpt_dir:  $CKPT_DIR"
echo "out_dir:   $OUT_DIR"
echo "adapters:  $ADAPTERS"
echo "seed:      $SEED"
echo "num_clips: $NUM_CLIPS"
echo "steps:     $NUM_STEPS"
echo ""

ARGS=(
  --config "$CONFIG"
  --mode generate
  --adapters "$ADAPTERS"
  --num-clips "$NUM_CLIPS"
  --seed "$SEED"
  --num-inference-steps "$NUM_STEPS"
  --out-dir "$OUT_DIR"
)
if [[ -n "$CLIP_INDICES" ]]; then
  ARGS+=(--clip-indices "$CLIP_INDICES")
fi

# enable_xformers=false：xformers flash-attention 在生成阶段的时空注意力形状下会
# 触发 "invalid configuration argument"，禁用后回退到稳健的 PyTorch SDPA。
python -m motion_proj.eval.evaluate "${ARGS[@]}" \
  "model.pretrained=$MODEL_PRETRAINED" \
  "model.enable_xformers=false" \
  "work_dir=$WORK_DIR" \
  "paths.ckpt_dir=$CKPT_DIR"

echo ""
echo "== Adapter evaluation complete =="
echo "Summary: $OUT_DIR/summary.json"
if command -v python >/dev/null 2>&1 && [[ -f "$OUT_DIR/summary.json" ]]; then
  python - "$OUT_DIR/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
rank = s.get("ranking", {})
print("按静态漂移排序:", rank.get("by_static_drift"))
print("推荐 checkpoint:", rank.get("recommended_checkpoint"))
for name, blob in s.get("adapters", {}).items():
    a = blob["aggregate"]
    def f(x): return "n/a" if x is None else f"{x:.4f}"
    print(f"  {name:20s} drift={f(a.get('static_drift_mean'))} "
          f"lpips={f(a.get('lpips_mean'))} psnr={f(a.get('psnr_mean'))} "
          f"ssim={f(a.get('ssim_mean'))}")
PY
fi
