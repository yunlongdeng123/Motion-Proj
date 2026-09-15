#!/usr/bin/env bash
set -euo pipefail
N=/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1
E=/root/autodl-tmp/envs/splatad-impact
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple PIP_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
"$E/bin/python" -m pip install -c "$N/scripts/splatad_runtime_constraints.txt" appdirs 'tyro==0.8.14' gdown h5py msgpack msgpack_numpy 'nerfacc==0.5.2' 'splines==0.3.0' pathos pytorch-msssim 'timm==0.6.7' 'protobuf==3.20.3' 'wandb==0.16.6' 'fastapi==0.110.0' pyngrok python-socketio zod 'dataclass-wizard==0.30.0' transforms3d websockets nodeenv yourdfpy pyliblzfse
"$E/bin/python" -m pip install --no-deps --no-build-isolation -e /root/autodl-tmp/external/worldsim_simimpact/pandaset-devkit/python
