"""核对真实输入和源码合同；只用已保存相机元数据，不初始化原生 CUDA Scene。"""

from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import json
import sys
import tempfile

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--V1-cameras', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    campaign = options.campaign.resolve()
    sys.path.insert(0, str(campaign/'engine'))
    from build_v4_campaign import strip_hooks
    from utils.v4_road import create_road_training, load_bundle, classify_units
    from utils.graphics_utils import getProjectionMatrixCenterShift
    from omegaconf import OmegaConf
    plan = json.loads((campaign/'plan.json').read_text())
    source = Path(plan['V1_source'])
    if strip_hooks((campaign/'engine/train.py').read_text()) != (source/'train.py').read_text():
        raise AssertionError('V1 训练源码不一致')
    checked = []
    for path in source.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == 'train.py':
            continue
        relative = path.relative_to(source)
        if path.read_bytes() != (campaign/'engine'/relative).read_bytes():
            raise AssertionError('原 V1 文件被修改：'+str(relative))
        checked.append(str(relative))
    for path in list((campaign/'tools').glob('*.py')) + list((campaign/'tests').glob('*.py')) + [
            campaign/'engine/train.py', campaign/'engine/utils/v4_road.py', campaign/'engine/utils/v4_road_render.py']:
        ast.parse(path.read_text(), filename=str(path))
    meta, arrays = load_bundle(plan['road_bundle'])
    observed, prior = classify_units(arrays)
    native_cameras = json.loads(options.V1_cameras.read_text())
    cameras, errors = [], []
    for camera in native_cameras:
        c2w = np.eye(4)
        c2w[:3, :3] = camera['rotation']; c2w[:3, 3] = camera['position']
        view = torch.as_tensor(np.linalg.inv(c2w).T, dtype=torch.float32)
        expected = np.asarray(meta['frames'][camera['img_name']]['world_to_camera']) @ np.linalg.inv(meta['world_to_model'])
        expected[:3] *= meta['meter_to_model']
        errors.append(float(np.abs(np.linalg.inv(c2w)-expected).max()))
        scale = .25
        projection = getProjectionMatrixCenterShift(.01, 1000., camera['cx']*scale, camera['cy']*scale,
            camera['fx']*scale, camera['fy']*scale, 480, 320).T
        cameras.append(SimpleNamespace(image_name=camera['img_name'],
            image_path='/metadata/image_0/'+camera['img_name']+'.png',
            fx=camera['fx']*scale, fy=camera['fy']*scale, cx=camera['cx']*scale, cy=camera['cy']*scale,
            image_width=480, image_height=320, world_view_transform=view, full_proj_transform=view@projection))
    baseline = json.loads(Path(plan['baseline_config']).read_text())
    native_config = source.parent.parent/'training_configs'/f'{source.name}_GN11.json'
    if json.loads(native_config.read_text()) != baseline:
        raise AssertionError('使用的 baseline 配置不是原 V1 原样配置')
    counts = {}
    with tempfile.TemporaryDirectory(prefix='cpu_bind_', dir=campaign) as temporary:
        for arm in ('B', 'R', 'RP'):
            config = json.loads((campaign/'configs'/f'{arm}.json').read_text())
            expected = dict(baseline, model_path=str(campaign/'arms'/arm/'run30k'),
                            v4_arm=arm, v4_bundle=plan['road_bundle'])
            if config != expected:
                raise AssertionError('三组配置未保持原 V1：'+arm)
            output = Path(temporary)/arm; output.mkdir()
            config['model_path'] = str(output)
            scene = SimpleNamespace(gaussians=SimpleNamespace(get_xyz=torch.empty(0, 3)),
                                    getTrainCameras=lambda: cameras)
            road = create_road_training(OmegaConf.create(config), scene)
            counts[arm] = 0 if road is None else len(road.layer.xyz)
            if road is not None and int(road.layer.steps) != 0:
                raise AssertionError('CPU 准备不能执行实际道路更新')
    if torch.cuda.is_initialized() or list((campaign/'arms').glob('*/run30k')):
        raise AssertionError('CPU 检查不得启动 CUDA 或真实实验')
    result = dict(status='PASSED_CPU_SOURCE_INPUT_AND_CONFIGURATION_CONTRACTS',
        V1_train_equal_after_removing_two_hooks=True, unchanged_V1_files=len(checked),
        native_V1_config_equals_baseline_copy=True,
        camera_metadata_frames=len(cameras), extrinsic_max_abs_error=max(errors),
        supported_road_points=int(arrays['surface_supported'].sum()), observed_units=int(observed.sum()),
        connected_prior_units=int(prior.sum()), road_allocated_per_arm=counts,
        actual_scene_training_updates=0, actual_scene_road_updates=0,
        CUDA_initialized=False, native_CUDA_rasterization_tested=False)
    options.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
