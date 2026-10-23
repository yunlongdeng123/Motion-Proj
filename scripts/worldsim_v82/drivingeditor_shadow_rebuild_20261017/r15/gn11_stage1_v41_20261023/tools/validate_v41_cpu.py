"""原 V1 文件/配置、真实49帧相机和密集数据绑定；零 CUDA/场景训练更新。"""
import argparse
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
torch.set_num_threads(1)
from v41_train_hooks import strip_hooks
from run_v41_arm import configuration


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--V1-cameras', type=Path, required=True)
    args = p.parse_args(); campaign = args.campaign.resolve()
    sys.path.insert(0, str(campaign/'engine'))
    from utils.graphics_utils import getProjectionMatrixCenterShift
    from utils.v4_road import load_bundle
    from utils.v41_road import create_road_training
    from omegaconf import OmegaConf
    from utils.paper_checkpoint import _validate_resume_configuration
    plan = json.loads((campaign/'plan.json').read_text())
    source = Path(plan['V1_source'])
    if strip_hooks((campaign/'engine/train.py').read_text()) != (source/'train.py').read_text():
        raise AssertionError('V1 train.py 未保持原样')
    checked = []
    for path in source.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == 'train.py': continue
        relative = path.relative_to(source)
        if path.read_bytes() != (campaign/'engine'/relative).read_bytes():
            raise AssertionError('原生文件被修改：'+str(relative))
        checked.append(str(relative))
    for root in ('tools', 'tests', 'engine/utils'):
        for path in (campaign/root).glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    meta, arrays = load_bundle(plan['road_bundle'])
    cameras = []
    for c in json.loads(args.V1_cameras.read_text()):
        c2w = np.eye(4); c2w[:3, :3] = c['rotation']; c2w[:3, 3] = c['position']
        view = torch.as_tensor(np.linalg.inv(c2w).T, dtype=torch.float32)
        projection = getProjectionMatrixCenterShift(.01, 1000., c['cx']*.25, c['cy']*.25,
            c['fx']*.25, c['fy']*.25, 480, 320).T
        cameras.append(SimpleNamespace(image_name=c['img_name'], image_path='/metadata/image_0/'+c['img_name']+'.png',
            fx=c['fx']*.25, fy=c['fy']*.25, cx=c['cx']*.25, cy=c['cy']*.25,
            image_width=480, image_height=320, world_view_transform=view, full_proj_transform=view@projection))
    counts = {}
    with tempfile.TemporaryDirectory(prefix='v41_cpu_bind_', dir=campaign) as tmp:
        for arm in ('R',):
            config = configuration(campaign, arm, 'train')
            config['model_path'] = str(Path(tmp)/arm); Path(config['model_path']).mkdir()
            scene = SimpleNamespace(gaussians=SimpleNamespace(get_xyz=torch.empty(0, 3)), getTrainCameras=lambda:cameras)
            training = create_road_training(OmegaConf.create(config), scene)
            if set(dict(training.layer.named_parameters())) != {'dc', 'opacity'}: raise AssertionError('参数预算变化')
            if int(training.layer.steps) != 0: raise AssertionError('CPU 准备执行了真实更新')
            counts[arm] = dict(points=len(training.layer.xyz), active=int(training.layer.active.sum()),
                dense_pixels=sum(int(row['visible'].sum()) for row in training.dense),
                ssim_windows=sum(int(row['ssim_valid'].sum()) for row in training.dense))
    smoke = configuration(campaign, 'R', 'smoke', 'split_R', 500)
    resumed = dict(configuration(campaign, 'R', 'smoke', 'split_R', 1000),
        start_checkpoint=str(campaign/'smoke/split_R/chkpnt500.pth'))
    _validate_resume_configuration(smoke, OmegaConf.create(resumed))
    if torch.cuda.is_initialized() or list((campaign/'arms').glob('*/run30k')):
        raise AssertionError('超出 GPU 启动前范围')
    result = dict(status='PASS_CPU_PRE_GPU', unchanged_V1_files=len(checked),
        V1_train_equal_after_removing_two_hooks=True, native_config_unchanged=True,
        native_strict_smoke_resume_config_passed=True,
        camera_frames=len(cameras), arms=counts, real_scene_training_updates=0,
        GPU_used=False, CUDA_initialized=False, CUDA_raster_backward_verified=False,
        real_lifecycle_smoke_passed=False, image_quality_improvement_verified=False)
    (campaign/'cpu_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
