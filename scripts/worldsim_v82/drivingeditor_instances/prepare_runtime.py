"""CPU-only: 独立复制已验收源码并绑定三个既有scene，绝不启动GPU或训练。"""
import argparse
import json
from pathlib import Path
import shutil


BASE = Path('/root/autodl-tmp/drivingeditor_repro')
MATRIX = BASE / 'campaign/hybrid30k_20261012'
FOLLOWUP = BASE / 'campaign/followup_scene_prep_20261013'


def copy_source(source, destination):
    if destination.exists():
        raise FileExistsError('独立source已存在，拒绝覆盖')
    destination.mkdir(parents=True)
    included = []
    for directory in ('scene', 'gaussian_renderer', 'utils', 'configs'):
        for path in (source / directory).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in ('.py', '.json', '.yaml', '.yml', '.md'):
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            included.append(str(path.relative_to(source)))
    for path in source.glob('*LICENSE*'):
        if path.is_file():
            shutil.copy2(path, destination / path.name)
            included.append(path.name)
    # 原渲染器和局部helper保留便于逐行审查；新的运行入口位于infra，不改原文件。
    return {'source': str(source), 'destination': str(destination), 'files': included, 'GPU_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', required=True, type=Path)
    parser.add_argument('--scene', choices=['gmc', 'silver', 'red'], default='gmc', help='当前授权只测GMC；其他scene仅保留既有准备')
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    scenes = [
        ('scene0017_000_048', MATRIX / 'frozen_repository', BASE / 'data/waymo_paper_scenes/scene0017_000_048',
         MATRIX / 'configs/GN11.json', MATRIX / 'runs/scene0017_000_048_hybrid_30000_seed0_GN11', 'scene0017_000_048_attempt2'),
        ('scene0008_023_071', FOLLOWUP / 'training_source/scene0008_023_071', FOLLOWUP / 'inputs/scene0008_023_071_hybrid_cpu_v1',
         FOLLOWUP / 'training_configs/scene0008_023_071_GN11.json', FOLLOWUP / 'runs/scene0008_023_071_hybrid_30000_seed0_GN11', 'scene0008_023_071'),
        ('scene0029_069_117', FOLLOWUP / 'training_source/scene0029_069_117', FOLLOWUP / 'inputs/scene0029_069_117_red_cpu_v1',
         FOLLOWUP / 'training_configs/scene0029_069_117_GN11.json', FOLLOWUP / 'runs/scene0029_069_117_hybrid_30000_seed0_GN11', 'scene0029_069_117')]
    receipts = []
    selected = {'silver': scenes[0], 'gmc': scenes[1], 'red': scenes[2]}[args.scene]
    if args.scene == 'gmc':
        selected = (*selected[:-1], 'scene0008_two_actor_v1')
    for scene, source, inputs, reference, run, prepared_name in [selected]:
        prepared = root / 'prepared' / prepared_name
        if not (prepared / 'instances.npz').is_file():
            raise FileNotFoundError('真实prepared缺失: ' + str(prepared))
        destination = root / 'source' / scene
        receipt = copy_source(source, destination)
        spec = {'schema': 'drivingeditor_rigid_scene_spec_v1', 'scene_id': scene, 'source_repository': str(destination),
                'source_scene': str(inputs), 'prepared': str(prepared), 'reference_config': str(reference),
                'background_checkpoint': str(run / 'chkpnt30000.pth'), 'environment_checkpoint': str(run / 'env_light_chkpnt30000.pth'),
                'matrix_tools': str(MATRIX / 'tools'), 'stage': 'fixed_GT_trajectory_frozen_background_baseline',
                'box_and_track_roles': 'explicit_training_priors', 'test_views': 'same49_training_reconstruction_no_heldout',
                'nonrigid_targets': 'unsupported_first_version', 'human_quality_verdict': None,
                'StreetCrafter_priority': True, 'scope': 'GMC reviewed two moving actors; parked vehicles retain original static branch'}
        spec_dir = root / 'specs'; spec_dir.mkdir(exist_ok=True)
        with (spec_dir / (scene + '.json')).open('x', encoding='utf-8') as stream:
            json.dump(spec, stream, ensure_ascii=False, indent=2)
        receipts.append(receipt)
    with (root / 'runtime_preparation.json').open('x', encoding='utf-8') as stream:
        json.dump({'status': 'ISOLATED_SOURCES_AND_REAL_SCENE_BINDINGS_PREPARED', 'scenes': receipts,
                   'old_assets_modified': False, 'Torch_imported': False, 'GPU_started': False}, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': 'ISOLATED_RUNTIME_PREPARED', 'scenes': [selected[0]]}))


if __name__ == '__main__': main()
