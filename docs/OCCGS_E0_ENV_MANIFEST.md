# OccGS E0 Environment Manifest

- created_at_utc: 2026-07-21T16:34:30Z
- status: smoke_passed
- env_path: /root/autodl-tmp/envs/drivestudio
- python: 3.9.25

## Sources (mirrors only; no download.pytorch.org / pypi.org)

- pip index: https://mirrors.aliyun.com/pypi/simple/
- torch wheels: https://mirrors.aliyun.com/pytorch-wheels/cu118/ (fallback: mirror.sjtu.edu.cn)
- conda channels: mirrors.tuna.tsinghua.edu.cn
- CUDA extensions: built from local clones under /root/autodl-tmp/third_party/{gsplat,pytorch3d,nvdiffrast}

## Pinned third_party commits

- drivestudio: e59bda4fa681f829dbb1d65f0de582b0f633c450 (https://github.com/ziyc/drivestudio.git)
- gsplat: 507f26e118afb78ef0224ad5e0d0701f1b973853 tag v1.3.0 (+ glm submodule)
- pytorch3d: 2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7 tag v0.7.5
- nvdiffrast: 253ac4fcea7de5f396371124af597e6cc957bfae

## Adapted stack (official requirements pin torch==2.0.0+cu117 was NOT used)

```
python 3.9.25
torch 2.1.2+cu118
torchvision 0.16.2+cu118
numpy 1.23.5
opencv 4.8.1
gsplat 1.3.0
pytorch3d 0.7.5
nvdiffrast 0.4.0
cuda_runtime 11.8
cuda_home /usr/local/cuda-11.8
gpu NVIDIA GeForce RTX 4090
driver 580.105.08
```

## Intentionally skipped (plan §5.5 / phase-1)

- xformers (official pin incompatible with adapted torch; not required for E0 smoke)
- smplx / SMPL human poses
- drivestudio-seg env (mask pipeline deferred)

## Disk

- data disk gate: keep >= 30 GiB free
- after E0 install: see live `df -h /root/autodl-tmp`
- drivestudio env size: ~6.8G
- motionproj / resim envs preserved untouched

## Smoke (E0)

- torch CUDA matmul: PASS
- gsplat import + forward/backward rasterization: PASS (peak ~8.8 MiB @ 128px / 1k gaussians)
- pytorch3d import: PASS
- nvdiffrast import: PASS
- DriveStudio configs/omnire.yaml parse: PASS

## Activate

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/drivestudio
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/third_party/drivestudio:$PYTHONPATH
```
