#!/bin/bash
# Frozen B0 reconstruction ladder (Street-Gaussians-style).
# Identical CLI overrides for S0/S1/S2 = B0-4 repeatability.
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/drivestudio
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/third_party/drivestudio:$PYTHONPATH
unset OMP_NUM_THREADS

OUT_ROOT=/root/autodl-tmp/runs/occgs_resim/b0_recon
DATA_ROOT=/root/autodl-tmp/data/occgs/processed_10Hz/mini
cd /root/autodl-tmp/third_party/drivestudio

# role scene_idx run_name
run_one() {
  local role=$1 sid=$2 name=$3
  local logdir=$OUT_ROOT/${name}
  mkdir -p "$logdir"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $name scene=$sid" | tee -a $OUT_ROOT/b0_ladder.log
  df -h /root/autodl-tmp | tee -a $OUT_ROOT/b0_ladder.log
  python tools/train.py \
    --config_file configs/streetgs.yaml \
    --output_root "$OUT_ROOT" \
    --project occgs_b0 \
    --run_name "$name" \
    dataset=nuscenes/3cams \
    data.data_root=$DATA_ROOT \
    data.scene_idx=$sid \
    data.start_timestep=0 \
    data.end_timestep=79 \
    data.pixel_source.load_smpl=False \
    data.pixel_source.test_image_stride=10 \
    data.preload_device=cpu \
    trainer.optim.num_iters=30000 \
    logging.saveckpt_freq=15000 \
    logging.vis_freq=5000 \
    logging.print_freq=1000 \
    render.render_novel=null \
    2>&1 | tee "$logdir/console.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE $name exit=$?" | tee -a $OUT_ROOT/b0_ladder.log
}

# B0-2: S0 3cams 8s
run_one S0 3 b0_2_s0_3cam8s
# B0-3: S1 dynamic
run_one S1 5 b0_3_s1_3cam8s
# B0-4: S2 with identical frozen config (repeatability)
run_one S2 4 b0_4_s2_3cam8s

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ALL_B0_DONE" | tee -a $OUT_ROOT/b0_ladder.log
