"""Prepare branch-control execution contracts only; never start GPU or a waiter."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.util
import json
import py_compile
import sys

import numpy as np
import torch

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = B / 'scene008_a5_branch_control_20261107_r1'
ENGINE = B / 'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
BASELINE = B / 'scene008_a5_support_ablation_20261106_r1/support'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    assert not path.exists(), path
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    assert not torch.cuda.is_initialized()
    config = json.loads((ROOT / 'config.json').read_text())
    base = json.loads((BASELINE / 'config.json').read_text())
    allowed = ['source_path', 'model_path', 'followup_scene_spec', 'test_iterations',
               'save_iterations', 'checkpoint_iterations', 'test_interval', 'paper_eval_train_each_checkpoint']
    diffs = {k: dict(baseline=base.get(k), candidate=config.get(k)) for k in set(base)|set(config) if base.get(k) != config.get(k)}
    assert set(diffs) <= set(allowed)
    expected = copy.deepcopy(base['followup_scene_spec'])
    expected['source_path'] = config['source_path']
    expected['actor_ply_paths'] = config['followup_scene_spec']['actor_ply_paths']
    assert expected == config['followup_scene_spec']
    assert config['test_iterations'] == [] and config['test_interval'] == 1000000000
    assert config['paper_eval_train_each_checkpoint'] is False
    assert config['save_iterations'] == config['checkpoint_iterations'] == [1000,5000]
    assert config['iterations'] == config['position_lr_max_steps'] == 30000
    audit = json.loads((ROOT / 'branch_control_cpu_audit.json').read_text())
    identity = json.loads((ROOT / 'identity_contract.json').read_text())
    assert audit['native_static_rows'] == 588123 and audit['native_dynamic_rows'] == 119810
    assert identity['dynamic_actor_aliases'] == [5,37] and identity['dynamic_source_real_points'] == [21358,98452]
    source = Path(config['source_path'])
    spec = importlib.util.spec_from_file_location('branch_mask_loader_cpu', ENGINE / 'utils/paper_masks.py')
    masks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(masks)
    coverage = []
    for raw in range(89,138):
        camera = SimpleNamespace(image_path=str(source / f'image_0/0008{raw:03d}.png'),
            image_width=480, image_height=320, world_view_transform=torch.zeros(1))
        md = masks.load_dynamic_mask(source, camera).numpy()[0].astype(bool)
        with np.load(B / f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz', allow_pickle=False) as data:
            a5 = data['actor_labels'] == 1
        coverage.append(dict(raw=raw, a5_pixels=int(a5.sum()), a5_dynamic_rgb_alpha_pixels=int((a5&md).sum()),
            a5_static_rgb_pixels=int((a5&~md).sum()), mask_sha256=hashlib.sha256(md.tobytes()).hexdigest()))
    total = sum(row['a5_pixels'] for row in coverage)
    static = sum(row['a5_static_rgb_pixels'] for row in coverage)
    assert total == 121128 and static == 0, (total, static)
    write(ROOT / 'actual_mask_loader_coverage.json', dict(status='PASS_ALL_FIXED_A5_PIXELS_IN_DYNAMIC_RGB_ALPHA_DOMAIN',
        frames=coverage, a5_fixed_pixels=121128, a5_dynamic_rgb_alpha_pixels=121128, a5_static_rgb_pixels=0,
        actual_loader=str(ENGINE / 'utils/paper_masks.py'), cuda_initialized=torch.cuda.is_initialized(), updates=0))
    engine_files = {str(p):dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)) for p in sorted(ENGINE.rglob('*.py'))}
    assert engine_files[str(ENGINE / 'gaussian_renderer/__init__.py')]['sha256'] == '35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
    assert engine_files[str(ENGINE / 'scene/gaussian_model.py')]['sha256'] == 'e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'
    inputs = [source / 'points3d.ply', source / 'points3d_dynamic_0.ply', source / 'points3d_dynamic_1.ply', source / 'transform_matrix.txt']
    inputs += sorted((source / 'dynamic_masks/dynamic_mask_0').glob('*.png'))
    inputs += sorted((source / 'dynamic_masks/shadow_mask_0').glob('*.png'))
    input_files = {str(p):dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)) for p in inputs}
    write(ROOT / 'engine_source_manifest.json', engine_files)
    write(ROOT / 'input_source_manifest.json', input_files)
    write(ROOT / 'execution_contract.json', dict(status='CPU_PREPARED_NOT_LAUNCHED_NOT_QUEUED',
        formal_stop=5000, phase_stops=[1000,5000], learning_rate_horizon=30000, seed=0,
        expected_initial_static_rows=588123, expected_initial_dynamic_rows=119810,
        dynamic_aliases=[5,37], dynamic_source_points=[21358,98452],
        native_original_config=str(BASELINE / 'config.json'), prepared_config_sha256=sha(ROOT / 'config.json'),
        exact_config_differences_from_fullsupport=diffs, engine_patch=False, loss_weight_changes={},
        evaluation_route='No automatic full49. Explicit first1k phase saves/pilots then exits; root views and explicitly invokes finish5k. Full49 requires matching completed pilot.',
        pilot_frames=[97,105,109], keep_only=True, delete_produced=False,
        native_delete_removes=[5,37], a37_only_delete_solution=False,
        rng='Original preserve_rng only initial metric import and checkpoint/evaluation nodes. Independent pilot process; no per-iteration CUDA RNG snapshot.',
        paired_comparison='1000/5000 camera, time_shift and LR match original fullsupport weight1; no cross-branch global-RNG/topology equality claim',
        scientific_boundary='Branch targets/priors/representation bundle; total2107 fewer rows=54duplicate+2053 original exclusion near-neighbors of unknown semantic identity; native KNN domain differs',
        first1k_command=f'{sys.executable} -B {ROOT}/source/run_branch_arm.py --phase first1k',
        finish5k_command=f'{sys.executable} -B {ROOT}/source/run_branch_arm.py --phase finish5k',
        full49_command=f'{sys.executable} -B {ROOT}/source/evaluate_branch_keep.py --model {ROOT}/formal5k --iteration 5000 --scope all49 --pilot-reference {ROOT}/keep_pilot_5k --output {ROOT}/keep_all49_5k',
        trigger='Root explicitly releases only after current static-RGB2x2 support5K review; this preparation never launches',
        no_waiter=True, GPU_used=False, actual_training_updates=0, shutdown_allowed=False, gpu_reset_allowed=False))
    scripts = sorted((ROOT / 'source').glob('*.py'))
    for path in scripts:
        py_compile.compile(str(path), cfile='/tmp/branch_'+hashlib.sha256(str(path).encode()).hexdigest()+'.pyc', doraise=True)
    write(ROOT / 'execution_source_manifest.json', {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in scripts})
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS_CPU_EXECUTION_PREPARATION_NOT_LAUNCHED', scripts=len(scripts),
        actual_mask_loader_dynamic_a5=121128, actual_mask_loader_static_a5=0, cuda_initialized=False, training_updates=0)))


if __name__ == '__main__':
    main()
