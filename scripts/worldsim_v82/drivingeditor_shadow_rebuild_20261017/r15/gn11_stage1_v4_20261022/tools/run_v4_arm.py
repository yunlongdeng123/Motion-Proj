"""三组同一 V1 配置入口。默认只显示配置，显式 --execute 才训练。"""

from pathlib import Path
import argparse
import json
import os
import sys
import time


def bind_scene_contract(training, config):
    """补接 V1 启动器的 sceneSpec；只检查并绑定元数据，不改 Scene/RNG。"""
    original_scene = training.Scene
    spec = config['followup_scene_spec']

    def checked_scene(*args, **kwargs):
        scene = original_scene(*args, **kwargs)
        cameras = sorted(scene.getTrainCameras(), key=lambda camera: float(camera.timestamp))
        identities = [Path(camera.image_path).parent.name+'/'+camera.image_name for camera in cameras]
        if identities != spec['frame_ids_with_camera'] or len(cameras) != 49:
            raise ValueError('actual Scene 身份与 V1 sceneSpec 不匹配')
        if (len(spec['model_timestamps']) != len(cameras)
                or not all(abs(float(camera.timestamp)-timestamp) < 1e-7
                           for camera, timestamp in zip(cameras, spec['model_timestamps']))):
            raise ValueError('actual Scene timestamps 与 V1 sceneSpec 不匹配')
        if not all(camera.image_width == 480 and camera.image_height == 320 for camera in cameras):
            raise ValueError('actual Scene render dimensions 不是 W480H320')
        if scene.getTestCameras(scale=4):
            raise ValueError('V1 固定49视图协议不允许新增 heldout')
        scene.followup_scene_spec = spec
        with (Path(config['model_path'])/'actual_scene_binding.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(status='PASS_ACTUAL_SCENE_BINDING',
                frame_ids=identities, timestamps=spec['model_timestamps'], render_wh=[480, 320],
                anchor_pose_frame=spec['anchor_frame']), stream, ensure_ascii=False, indent=2)
        return scene

    training.Scene = checked_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--arm', choices=('B', 'R', 'RP'), required=True)
    parser.add_argument('--execute', action='store_true')
    options = parser.parse_args()
    campaign = options.campaign.resolve()
    plan = json.loads((campaign/'plan.json').read_text(encoding='utf-8'))
    config = json.loads((campaign/'configs'/f'{options.arm}.json').read_text(encoding='utf-8'))
    baseline = json.loads(Path(plan['baseline_config']).read_text(encoding='utf-8'))
    expected = dict(baseline, model_path=str(campaign/'arms'/options.arm/'run30k'),
                    v4_arm=options.arm, v4_bundle=plan['road_bundle'])
    if config != expected or baseline['start_checkpoint'] is not None or baseline['iterations'] != 30000:
        raise ValueError('三组只能改变道路开关与输出目录，其余配置逐项相同')
    print(json.dumps(dict(arm=options.arm, output=config['model_path'], iterations=30000,
                         execute=options.execute, second_stage_updates=0), indent=2))
    if not options.execute:
        return
    output = Path(config['model_path'])
    if output.exists():
        raise FileExistsError('不覆盖/自动续训已有实验')
    engine = campaign/'engine'
    sys.path.insert(0, str(engine))
    import torch
    from omegaconf import OmegaConf
    import train
    from utils.general_utils import seed_everything
    if Path(train.__file__).resolve() != engine/'train.py':
        raise RuntimeError('错误的 V4 engine 导入')
    if not torch.cuda.is_available():
        raise RuntimeError('训练需要 CUDA；本入口不自动降级为 CPU 实验')
    bind_scene_contract(train, config)
    output.mkdir()
    (output/'effective_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    os.chdir(engine)
    torch.cuda.set_device(torch.device('cuda:0'))
    seed_everything(config['seed'])
    started = time.perf_counter()
    train.training(OmegaConf.create(config))
    (output/'v4_training_returned.json').write_text(json.dumps(dict(
        status='TRAINING_RETURNED', arm=options.arm, V1_updates=30000,
        road_updates=0 if options.arm == 'B' else 30000,
        training_wall_seconds=time.perf_counter()-started,
        wall_time_scope='training() including initialization, evaluation, checkpoint and road updates',
        second_stage_updates=0, quality_accepted=False, human_verdict=None), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
