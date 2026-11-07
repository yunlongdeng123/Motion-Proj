"""CPU-only preparation: register static RGB weight 5, preserve native engine/data."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np
import torch

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = CAMPAIGN / 'scene008_a5_static_rgb5_ablation_20261107_r2'
BASE = CAMPAIGN / 'scene008_a5_support_ablation_20261106_r1'
ENGINE = CAMPAIGN / 'r15_baseline_owner5_20261018/training_source/scene0008_089_137'
OBSERVATIONS = CAMPAIGN / 'scene008_instances_20261105_r1/observations/frames'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    assert not torch.cuda.is_initialized()
    assert (ENGINE / 'gaussian_renderer/__init__.py').is_file()
    assert sha(ENGINE / 'gaussian_renderer/__init__.py') == '35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
    assert sha(ENGINE / 'scene/gaussian_model.py') == 'e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'
    engine_files = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                    for p in sorted(ENGINE.rglob('*.py'))}
    mask_module = load_module('loss2x2_paper_masks_cpu', ENGINE / 'utils/paper_masks.py')
    runtime = load_module('loss2x2_runtime_cpu', ENGINE / 'utils/hybrid30k_runtime.py')
    losses = load_module('loss2x2_losses_cpu', ENGINE / 'utils/hybrid30k_losses.py')
    arms = {}
    for name, expected_static in [('control', 591672), ('support', 611588)]:
        arm = ROOT / name
        arm.mkdir(exist_ok=False)
        (arm / 'source').mkdir()
        baseline_path = BASE / name / 'config.json'
        base = json.loads(baseline_path.read_text())
        runtime.validate_config(base)
        assert base['hybrid_lambda_static_rgb'] == 1.0
        config = copy.deepcopy(base)
        config['hybrid_lambda_static_rgb'] = 5.0
        config['model_path'] = str(arm / 'formal1k')
        diff = {k: dict(baseline=base.get(k), candidate=config.get(k))
                for k in set(base) | set(config) if base.get(k) != config.get(k)}
        assert set(diff) == {'model_path', 'hybrid_lambda_static_rgb'}
        shutil.copy2(baseline_path, arm / 'baseline_config.json')
        shutil.copy2(BASE / name / 'identity_contract.json', arm / 'identity_contract.json')
        write(arm / 'config.json', config)
        source = Path(config['source_path'])
        coverage = []
        for raw in range(89, 138):
            camera = SimpleNamespace(image_path=str(source / 'image_0' / f'0008{raw:03d}.png'),
                                     image_width=480, image_height=320,
                                     world_view_transform=torch.zeros(1))
            md = mask_module.load_dynamic_mask(source, camera).numpy()[0].astype(bool)
            with np.load(OBSERVATIONS / f'{raw:03d}.npz', allow_pickle=False) as data:
                a5 = np.ascontiguousarray(data['actor_labels'] == 1)
            overlap = int((a5 & md).sum())
            assert overlap == 0
            coverage.append(dict(raw=raw, a5_pixels=int(a5.sum()),
                                 a5_pixels_in_static_complement=int((a5 & ~md).sum()),
                                 a5_pixels_excluded_by_dynamic_mask=overlap,
                                 dynamic_pixels=int(md.sum()),
                                 a5_mask_sha256=hashlib.sha256(a5.tobytes()).hexdigest(),
                                 dynamic_mask_sha256=hashlib.sha256(md.tobytes()).hexdigest()))
        assert sum(r['a5_pixels'] for r in coverage) == 121128
        critical = [source / 'points3d.ply', source / 'points3d_dynamic_0.ply', source / 'transform_matrix.txt']
        critical += sorted((source / 'dynamic_masks/dynamic_mask_0').glob('*.png'))
        critical += sorted((source / 'dynamic_masks/shadow_mask_0').glob('*.png'))
        source_files = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                        for p in critical}
        contract = dict(arm=name, experiment='static_rgb_supervision_strength_2x2',
                        formal_stop=5000, phase_stops=[1000, 5000], learning_rate_horizon=30000,
                        expected_initial_static_rows=expected_static, expected_initial_dynamic_rows=98452,
                        source_input=str(source), baseline_root=str(BASE / name), engine=str(ENGINE),
                        renderer_sha256=sha(ENGINE / 'gaussian_renderer/__init__.py'),
                        gmodel_sha256=sha(ENGINE / 'scene/gaussian_model.py'),
                        source_config_sha256=sha(baseline_path), fresh=True, seed=0,
                        added_static_support_rows=0 if name == 'control' else 19916,
                        method_changes={'hybrid_lambda_static_rgb': {'baseline': 1.0, 'candidate': 5.0}},
                        exact_config_differences=diff, baseline_native_validator='PASS_UNMODIFIED_BASE_CONFIG',
                        candidate_registration='Exact original baseline except static RGB weight and output path',
                        no_engine_patch=True, dynamic_aliases=[37], a5_branch='static',
                        delete_semantics='native render_s deletes the complete selected a37 dynamic pool',
                        shutdown_allowed=False, gpu_reset_allowed=False)
        write(arm / 'arm_contract.json', contract)
        write(arm / 'input_source_manifest.json', source_files)
        write(arm / 'static_target_coverage.json', dict(status='PASS_ALL_FIXED_A5_PIXELS_IN_NATIVE_STATIC_RGB_TARGET',
              frame_count=49, a5_pixels=121128, excluded_a5_pixels=0, frames=coverage,
              native_mask_function=str(ENGINE / 'utils/paper_masks.py'), mask_union_includes_existing_shadow=True,
              normalization='L1 mean over 3*320*480; SSIM whole masked image; no foreground pixel normalization'))
        for filename in ['train_loss_arm.py', 'run_loss_arm.py']:
            shutil.copy2(ROOT / 'source' / filename, arm / 'source' / filename)
        shutil.copy2(BASE / name / 'source/evaluate_a5.py', arm / 'source/evaluate_a5.py')
        assert sha(arm / 'source/evaluate_a5.py') == 'be18a16926f8ebebe345dd7e1ae5cf6ce80378efb2fa6c43ed7ab3271ecaf1dc'
        arms[name] = dict(root=str(arm), contract=contract, coverage=str(arm / 'static_target_coverage.json'),
                          input_files=source_files)

    # Test the actual composer on deterministic CPU tensors, without scene/CUDA loading.
    config = json.loads((ROOT / 'control/config.json').read_text())
    rgb = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    state = {key: torch.tensor(.2, dtype=torch.float64) for key in
             ['global_rgb', 'dynamic_rgb', 'sky_rgb', 'sky', 'depth', 'inverse_depth_smoothness', 'velocity_smoothness']}
    state.update(static_rgb=rgb, iteration=1,
                 dynamic_alpha=torch.full((1, 3, 4), .2, dtype=torch.float64),
                 dynamic_mask=torch.zeros((1, 3, 4), dtype=torch.float64),
                 static_t_map=torch.full((1, 3, 4), .3, dtype=torch.float64),
                 static_v_map=torch.full((3, 3, 4), .1, dtype=torch.float64),
                 alpha=torch.full((1, 3, 4), .7, dtype=torch.float64),
                 nps_features=torch.zeros((3, 3, 4), dtype=torch.float64),
                 nps_distances=torch.zeros((1, 3, 4), dtype=torch.float64))
    config1 = dict(config, hybrid_lambda_static_rgb=1.0)
    nps_fn = lambda *args: rgb * 0 + .4
    result1 = losses.compose_hybrid_loss(state, config1, nps_fn=nps_fn)
    result5 = losses.compose_hybrid_loss(state, config, nps_fn=nps_fn)
    gradient1 = torch.autograd.grad(result1.total, rgb, retain_graph=True)[0]
    gradient5 = torch.autograd.grad(result5.total, rgb)[0]
    assert abs(float(result5.total - result1.total) - 4 * float(rgb)) < 1e-12
    assert float(gradient1) == 1.0 and float(gradient5) == 5.0
    for key in result1.weighted:
        if key != 'static_rgb':
            assert torch.equal(result1.weighted[key], result5.weighted[key])
    assert not torch.cuda.is_initialized()
    cpu = dict(status='PASS_NATIVE_COMPOSER_ONE_WEIGHT_ONLY_CPU', optimization_updates=0,
               cuda_initialized=False, total_delta=float(result5.total - result1.total),
               expected_delta=4 * float(rgb), static_scalar_gradient1=float(gradient1),
               static_scalar_gradient5=float(gradient5), all_other_weighted_terms_identical=True,
               composer=str(ENGINE / 'utils/hybrid30k_losses.py'), composer_sha256=sha(ENGINE / 'utils/hybrid30k_losses.py'))
    write(ROOT / 'engine_source_manifest.json', engine_files)
    write(ROOT / 'cpu_validation.json', cpu)
    contract = dict(status='PREPARED_CPU_NO_GPU_STARTED', created_unix=time.time(), root=str(ROOT), arms=arms,
                    comparison='Initialization(original versus a5 full-source support) x static RGB multiplier(1 versus 5)',
                    native_static_rgb='(.8*L1(Is*(1-Md),GT*(1-Md))+.2*(1-SSIM(Is*(1-Md),GT*(1-Md))))',
                    native_dynamic_rgb='5*(.8*L1(Id*Md,GT*Md)+.2*(1-SSIM(Id*Md,GT*Md)))',
                    global_rgb_weight=1.0, static_rgb_weights=[1.0, 5.0], dynamic_rgb_weight=5.0,
                    nominal_global_plus_branch_rgb_coefficient_static='2 -> 6; distinct renders/SSIM prevent simple gradient equivalence',
                    identifiable_hypothesis='Effect of stronger whole-complement static RGB at each fixed initialization, including their interaction',
                    limitations=['Static RGB includes all complementary pixels/background; not a5-local intervention',
                                 'Does not equalize dynamic opacity, time/velocity priors, time_shift behavior, or densification',
                                 'Support adds local density/capacity and can change native KNN initialization scales',
                                 'Single seed and observed-view reconstruction; no heldout or statistical causal proof'],
                    fixed_other_loss_weights={k:v for k,v in config.items() if k.startswith('lambda_') or k.startswith('paper_lambda') or k.startswith('hybrid_lambda')},
                    fixed_static_regularization={'duration': '.1 / mean(abs(static_duration_feature/combined_alpha))',
                                                'velocity': '.01 * mean(abs(static_velocity_feature/combined_alpha))'},
                    fixed_dynamic_opacity='100*mean((raw dynamic_alpha-Md)^2), whole H*W',
                    fixed_topology='Original static/dynamic thresholds, reset, SGCM, SH progression; original known radius bookkeeping retained',
                    mask_domain='49 raw089-137; all121128 fixed a5 pixel-frames are in (1-Md)',
                    lifetime='Fresh0->1000 complete save/native eval -> independent all49 A5 evaluation -> exact full-state restore at1000 ->5000 complete save/eval -> independent all49 A5 evaluation',
                    formal_updates_per_new_arm=5000, additional_engineering_training_updates=0,
                    historical_fresh5k_wall_seconds={'control': 899.3865345604718, 'support': 912.367506487295},
                    smoke_policy='1K real training/save/eval/complete restore is first lifecycle; if measured effective5K estimate approaches1h, stop for long-smoke handling before release',
                    no_engine_or_existing_input_modifications=True, shutdown_allowed=False, gpu_reset_allowed=False,
                    cpu_validation=cpu, architecture='outputs/a5_regression_20261106/architecture.svg')
    write(ROOT / 'loss2x2_contract.json', contract)
    print(json.dumps(dict(status=contract['status'], root=str(ROOT), static_counts=[591672,611588],
                          a5_mask_pixels=121128, excluded_a5_pixels=0, cuda_initialized=False,
                          config_differences=['hybrid_lambda_static_rgb','model_path']), ensure_ascii=False))


if __name__ == '__main__':
    main()
