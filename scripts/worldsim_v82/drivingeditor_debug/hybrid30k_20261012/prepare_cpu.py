"""生成隔离源码、30k四臂配置与局部diff；不导入renderer或加载模型。"""
import ast
import difflib
import json
import os
from pathlib import Path
import shutil

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/drivingeditor_repro')
SOURCE = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
ROOT = BASE / 'campaign/hybrid30k_20261012'
PARENT = BASE / 'runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline'


def arm_config(original, arm):
    cfg = dict(original)
    for key in list(cfg):
        if key.startswith('diagnostic_') or key in ('paper_lambda_dynamic_opacity', 'paper_lambda_dynamic_rgb'):
            del cfg[key]
    cfg.update(iterations=30000, position_lr_max_steps=30000, start_checkpoint=None,
               checkpoint_iterations=[10000, 20000, 30000], save_iterations=[10000, 20000, 30000],
               test_iterations=[10000, 20000, 30000], seed=0, eval=False,
               model_path=str(ROOT / 'runs' / arm['run_name']),
               paper_method=True, hybrid_method=True, paper_eval_train_each_checkpoint=True,
               hybrid_nps_enabled=arm['hybrid_nps_enabled'], paper_sgcm_enabled=arm['paper_sgcm_enabled'],
               sgcm_guard_enabled=True, hybrid_lambda_dynamic_rgb=5.0, hybrid_lambda_sky_rgb=5.0,
               hybrid_lambda_static_rgb=1.0, hybrid_lambda_dynamic_opacity=100.0,
               hybrid_time_reg_eps=1e-8, hybrid_gradient_nodes=[1, 1000, 10000, 15000, 20000, 30000],
               hybrid_opacity_reset_policy='densification_only', hybrid_arm=arm['arm'],
               paper_lambda_nps=.01, paper_lambda_sky=.05, paper_lambda_depth=.1,
               lambda_t_reg=.1, lambda_v_reg=.01, lambda_inv_depth=.1, lambda_v_smooth=.1,
               lambda_opacity_entropy=.05, lambda_dssim=.2, lidar_decay=1.0,
               paper_opacity_domain='full raw-alpha MSE against binary Md; coefficient100 inherited control',
               paper_weight_provenance='2024 active weights retained; Eq7=.05, Eq16=.1, NPS=.01 are declared unverified author coefficients',
               resolution_scales=[4], resolution=-1)
    return cfg


def ignore_assets(directory, names):
    forbidden = {'.git', '__pycache__', '.pytest_cache', 'runs', 'outputs', 'build', 'dist'}
    ignored = []
    for name in names:
        path = Path(directory) / name
        if name in forbidden or path.is_symlink() or '.codexbak' in name:
            ignored.append(name)
        elif path.is_file() and path.suffix in ('.pth', '.pt', '.npz', '.npy', '.png', '.jpg', '.mp4', '.so', '.pyc'):
            ignored.append(name)
        elif path.is_file() and path.stat().st_size > 16 * 1024**2:
            raise ValueError('拒绝复制大型非源码资产: ' + str(path))
    return ignored


def main():
    import train_patch
    import sgcm_patch
    from runtime import validate_config
    target = ROOT / 'frozen_repository'
    if target.exists():
        raise FileExistsError('目标已存在，拒绝覆盖已准备源码')
    original = json.loads((PARENT / 'checkpoint_events/iteration_030000.json').read_text())['config']
    matrix = json.loads((HERE / 'experiment_matrix.json').read_text(encoding='utf-8'))
    source_files = {'train.py': train_patch.patch_train_source,
                    'utils/paper_checkpoint.py': train_patch.patch_checkpoint_source,
                    'scene/paper_sgcm.py': sgcm_patch.patch_sgcm_source,
                    'scene/gaussian_model.py': sgcm_patch.patch_gaussian_source}
    patched = {}
    for name, patch in source_files.items():
        old = (SOURCE / name).read_text(encoding='utf-8').replace('\r\n', '\n')
        new = patch(old)
        ast.parse(new)
        patched[name] = old, new
    configs = {arm['arm']: arm_config(original, arm) for arm in matrix['arms']}
    for cfg in configs.values():
        validate_config(cfg)
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE, target, ignore=ignore_assets)
    changes = []
    for name, (old, new) in patched.items():
        (target / name).write_text(new, encoding='utf-8')
        changes.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                           fromfile='frozen/' + name, tofile='hybrid/' + name))
    helpers = {'losses.py': 'utils/hybrid30k_losses.py', 'runtime.py': 'utils/hybrid30k_runtime.py',
               'metrics.py': 'utils/hybrid30k_metrics.py', 'sgcm_helpers.py': 'scene/hybrid_sgcm_helpers.py'}
    helpers['static_time.py'] = 'utils/hybrid30k_static_time.py'
    for name, relative in helpers.items():
        source = HERE / name
        if not source.is_file():
            raise FileNotFoundError('局部helper缺失: ' + str(source))
        shutil.copyfile(source, target / relative)
        text = source.read_text(encoding='utf-8')
        changes.extend(difflib.unified_diff([], text.splitlines(True), fromfile='/dev/null', tofile='hybrid/' + relative))
    (ROOT / 'source_delta.patch').write_text(''.join(changes), encoding='utf-8')
    (ROOT / 'configs').mkdir(exist_ok=True)
    for arm, cfg in configs.items():
        (ROOT / 'configs' / (arm + '.json')).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    preparation = {'status': 'sources_materialized_GPU_not_started',
                   'task_id': 'WS-V82-DRIVINGEDITOR-HYBRID30K-20261012', 'run_id': 'r1',
                   'repository': str(target), 'arms': list(configs), 'iterations_each': 30000,
                   'iterations_total': 120000, 'checkpoint_iterations': [10000, 20000, 30000],
                   'old_SV_plan': 'retained, superseded, not queued',
                   'GPU_started': False, 'checkpoint_loaded': False,
                   'changed_files': list(source_files), 'added_files': list(helpers.values()),
                   'failure_ledger_refs': ['V82-F01', 'V76-F03'], 'failure_ledger_delta': 'none',
                   'human_verdict': None}
    (ROOT / 'preparation.json').write_text(json.dumps(preparation, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(preparation, ensure_ascii=False))


if __name__ == '__main__':
    main()
