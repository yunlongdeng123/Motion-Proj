"""Prepared single-variable outside-opacity30k->30300 hypothesis. Default CPU only.

Run requires the root's exact prepared-plan receipt after GPU ownership handoff.
The approved original harness remains unchanged. Only the bound replica has the
two reviewed hypothesis files; reset, SGCM, renderer and original paper terms stay.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

BASE = Path('/root/autodl-tmp/drivingeditor_repro')
PARENT = BASE / 'runs/scene0017_000_048_paper_50000_seed0_resume_vgpu1011_baseline'
REPO = BASE / 'campaign/resume_vgpu_baseline_20261011/frozen_repository'
PLAN = BASE / 'campaign/vgpu_bugfix_20261012/component_diagnostic/continuation_plan'
VARIANT_PLAN = PLAN / 'opacity_outside_prepared_plan.json'
BASELINE = BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic/original30k_to30300'
FIRST = 30001
STOP = 30300
NODES = (30001, 30010, 30050, 30100, 30200, 30300)
FRAME_IDS = ['image_0/0017' + str(i).zfill(3) for i in (0, 10, 20, 30, 48)]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


def prepare_original(output):
    import torch
    from omegaconf import OmegaConf
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU prepare initialized CUDA')
    output = output.resolve()
    expected_parent = BASE / 'campaign/vgpu_bugfix_20261012/continuation_diagnostic'
    if output.parent != expected_parent or output.exists():
        raise ValueError('New isolated output directly under continuation_diagnostic required')
    event_path = PARENT / 'checkpoint_events/iteration_030000.json'
    event = json.loads(event_path.read_text())
    state = torch.load(PARENT / 'training_state30000.pth', map_location='cpu', weights_only=False)
    effective = OmegaConf.to_container(OmegaConf.load(PARENT / 'effective_config.yaml'), resolve=True)
    if event['iteration'] != 30000 or state['iteration'] != 30000 or event['status'] != 'complete' or state['config'] != event['config'] or effective != event['config']:
        raise ValueError('CPU parent complete bundle/config mismatch')
    cfg = OmegaConf.create(event['config'])
    cfg.model_path = str(output)
    cfg.start_checkpoint = str(PARENT / 'chkpnt30000.pth')
    current = OmegaConf.to_container(cfg, resolve=True)
    changed = [key for key in sorted(current) if current[key] != event['config'][key]]
    if changed != ['model_path', 'start_checkpoint'] or cfg.iterations != 50000 or cfg.position_lr_max_steps != 50000 or cfg.seed != 0:
        raise ValueError('Diagnostic changed original horizon/seed/schedule')
    pins = {}
    for branch, row in event['checkpoints'].items():
        path = Path(row['path']).resolve()
        if path.parent != PARENT or path.stat().st_size != row['bytes']:
            raise ValueError('Missing or truncated saved30k component')
        pins[str(path)] = sha(path)
    audit = json.loads((PLAN.parent / 'cpu_audit.json').read_text())
    for path, row in audit['original_files'].items():
        if Path(path).suffix in ('.pth',) and not path.endswith('30000.pth'):
            continue
        if sha(path) != row['sha256']:
            raise ValueError('Source/kernel/data/checkpoint changed since inference audit: ' + path)
        pins[path] = row['sha256']
    runtime = json.loads((PARENT / 'runtime_environment.json').read_text())
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    reference_events = sorted(PARENT.glob('events.out.tfevents.*'))
    if len(reference_events) != 1:
        raise ValueError('Ambiguous formal-child TensorBoard reference')
    reference_path = reference_events[0]
    accumulator = EventAccumulator(str(reference_path), size_guidance={'scalars': 0})
    accumulator.Reload()
    references = {}
    for tag in accumulator.Tags()['scalars']:
        if not tag.startswith('train/') or 'loss_viewpoint' in tag.lower() or not any(key in tag.lower() for key in ('loss', 'psnr')):
            continue
        values = [event.value for event in accumulator.Scalars(tag) if event.step == FIRST]
        if len(values) != 1:
            raise ValueError('Missing or duplicated first-step TensorBoard reference: ' + tag)
        references[tag.removeprefix('train/')] = float(values[0])
    required_reference = {'loss', 'psnr', 'loss_paper_global_rgb', 'loss_paper_dynamic_rgb', 'loss_paper_sky', 'loss_paper_nps', 'loss_paper_dynamic_opacity', 'loss_paper_depth'}
    if not required_reference.issubset(references):
        raise ValueError('First-step loss/PSNR/six paper-component TensorBoard reference is incomplete')
    pins[str(reference_path)] = sha(reference_path)
    plan = {'schema_version': 1, 'status': 'prepared_not_started', 'parent': str(PARENT), 'parent_iteration': 30000, 'first_iteration': FIRST, 'stop_iteration': STOP, 'steps': STOP - FIRST + 1, 'horizon': 50000, 'position_lr_max_steps': 50000, 'seed': 0, 'output': str(output), 'repository': str(REPO), 'training_config_changed_fields': changed, 'training_config': current, 'original_file_sha256': pins, 'harness_sha256': sha(__file__), 'observer_sha256': sha(PLAN / 'runtime_safety_parameterized.py'), 'extra_eval_nodes': list(NODES), 'early_fixed_camera_ids': FRAME_IDS, 'terminal_eval_camera_count': 49, 'first_step_reference': {'source_path': str(reference_path), 'source_sha256': pins[str(reference_path)], 'iteration': FIRST, 'values': references, 'comparison_policy': 'Report actual values, exact equality and absolute deltas; no imposed or relaxed tolerance.'}, 'allocator_environment_original': runtime.get('pytorch_cuda_alloc_conf', ''), 'original_runtime': runtime, 'training_math': 'Import and call frozen original training.training(cfg): full original loss, backward, Adam, reset and SGCM rules. Observer stops only after30300 post-step complete_eval hook; args.iterations remains50000.', 'restore_policy': 'Exact four saved30k files; all saved parameters, conditional Adam steps/moments/options/device placement, all RNG, sampler/order, scale, model runtime, EMA and SGCM geometry/metadata. Parameterized copy of existing restore observer; no seed after restore.', 'extra_eval_rng_policy': 'Capture Python/NumPy/TorchCPU/allCUDA RNG before each extra diagnostic eval; restore in finally; hash-verify equality after. No sampling shuffle, no new seed, no Scene reload or camera reconstruction.', 'runtime_import_verification': 'Verify imported training/renderer/Gaussian/Camera/environment/guard module paths and actual rasterizer Python+binary paths/SHA against CPU pinned originals before training.training.', 'diagnostics': 'Every true loss/camera/time_shift/LR/EMA logged each step; full parameter/optimizer finite checks first,node,last; static/dynamic activated opacity percentiles and shared raw cubemap statistics at eval nodes; early fixed5 original cameras, terminal30300 all49 original cameras: PNGs gt/composite/static/dynamic plus raw alpha/environment/foreground/depth NPZ and mask region statistics.', 'stop_policy': 'Raise own DiagnosticComplete observer exception after true30300 full backward and Adam; no formal event published. Save isolated terminal30300 static/dynamic/environment + exact continuation state after RNG-safe final49 evaluation; hash and CPU-load validate all four, publish diagnostic_snapshot_manifest.json explicitly separate from formal events.', 'algorithm_changes': [], 'allow_training': False, 'GPU_ownership': 'Acquire same gpu_slot.lock nonblocking; require idle compute before restore; root must first release inference ownership and provide exact plan receipt.'}
    return plan, cfg, state


def load_experiment(binding_path):
    binding_path = Path(binding_path).resolve()
    binding = json.loads(binding_path.read_text())
    guard_path = Path(binding['guard_path']).resolve()
    if sha(guard_path) != binding['guard_sha256']:
        raise RuntimeError('Actual hypothesis guard SHA differs from binding')
    spec = importlib.util.spec_from_file_location('bound_opacity_outside_guard', guard_path)
    guard_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_module)
    if Path(guard_module.__file__).resolve() != guard_path:
        raise RuntimeError('Actual hypothesis guard import path differs')
    return binding, guard_module.load_guard(binding_path)


def prepare(output, binding_path):
    import torch
    from omegaconf import OmegaConf
    binding, hypothesis_guard = load_experiment(binding_path)
    if Path(output).resolve() != Path(binding['output']).resolve():
        raise ValueError('Output must equal exactly bound isolated hypothesis output')
    plan, original_cfg, state = prepare_original(output)
    checks = {'sources': hypothesis_guard.verify_sources(), 'protected': hypothesis_guard.verify_protected_files(), 'auxiliary': hypothesis_guard.verify_auxiliary_files()}
    cfg = OmegaConf.create(binding['training_config'])
    current = OmegaConf.to_container(cfg, resolve=True)
    if {k: v for k, v in current.items() if k not in binding['added_training_fields']} != plan['training_config']:
        raise ValueError('Candidate differs from original baseline config beyond bound added fields')
    cpu_spec = importlib.util.spec_from_file_location('isolated_guard_CPU_validation', Path(binding['repository']) / 'utils/paper_checkpoint.py')
    cpu_guard = importlib.util.module_from_spec(cpu_spec)
    cpu_spec.loader.exec_module(cpu_guard)
    checks['configuration'] = hypothesis_guard.validate_config(state['config'], current, cpu_guard._validate_resume_configuration)
    if binding['evaluation'] != {'nodes': list(NODES), 'early_fixed_camera_ids': FRAME_IDS, 'terminal_eval_camera_count': 49, 'all_extra_eval_rng_preserved': True}:
        raise ValueError('Candidate changed approved camera/evaluation schedule')
    baseline_status = json.loads((BASELINE / 'status.json').read_text())
    baseline_report = json.loads((BASELINE / 'continuation_diagnostic.json').read_text())
    if baseline_status['status'] != 'complete' or baseline_status['actual_exit_code'] != 0 or baseline_status['gpu_lock_held'] or not baseline_report['terminal_full49_evaluation']:
        raise RuntimeError('Complete original baseline49/exit/released lock prerequisite failed')
    baseline_rows = [json.loads(line) for line in Path(binding['all_iteration_pair_reference']['source_path']).read_text().splitlines()]
    reference_rows = [{key: row[key] for key in ('iteration', 'camera_id', 'time_shift', 'learning_rates')} for row in baseline_rows]
    if reference_rows != binding['all_iteration_pair_reference']['rows'] or [row['iteration'] for row in reference_rows] != list(range(FIRST, STOP + 1)):
        raise RuntimeError('Bound all300 sampling/LR reference differs from actual baseline')
    if {key: baseline_rows[0]['losses'][key] for key in binding['first_step_pair_reference']['values']} != binding['first_step_pair_reference']['values']:
        raise RuntimeError('Bound first10 scalar reference differs from actual original baseline')
    original_pins = dict(plan['original_file_sha256'])
    original_pins.update(binding['protected_original_file_sha256'])
    original_pins.update(binding['auxiliary_file_sha256'])
    original_pins[str(Path(binding_path).resolve())] = sha(binding_path)
    # Independent CPU inventory covers all49 calib/pose/RGB/sky/actor and all467
    # saved input files. No independent per-frame Scene pose receipt was stored.
    common_path = BASE / 'campaign/vgpu_bugfix_20261012/sgcm_data_audit/outside_opacity_review/common49_input_manifest_review_cpu.json'
    common = json.loads(common_path.read_text())
    if common['status'] != 'PASS' or not common['all_original_input_sha_match'] or common['full_input_file_count'] != 467 or sha(common['input_manifest']) != common['input_manifest_sha256']:
        raise RuntimeError('Complete saved input manifest evidence failed')
    if not common['new_shadow_mask_absent'] or (Path(common['data_source']) / 'dynamic_masks/shadow_mask_0').exists():
        raise RuntimeError('Unbound new appearance shadow-mask directory exists')
    original_pins[str(common_path)] = sha(common_path)
    for relative, expected in common['file_sha256'].items():
        path = Path(common['data_source']) / relative
        if not path.resolve().is_relative_to(Path(common['data_source']).resolve()) or sha(path) != expected:
            raise RuntimeError('Actual complete input inventory differs: ' + relative)
        original_pins[str(path)] = expected
    final_eval_path = BASELINE / 'extra_eval/30300/diagnostic_eval.json'
    baseline_final = json.loads(final_eval_path.read_text())
    if len(baseline_final['frames']) != 49 or not baseline_final['all_rng_unchanged']:
        raise RuntimeError('Actual baseline49 evaluation reference incomplete')
    original_pins[str(final_eval_path)] = sha(final_eval_path)
    for frame in baseline_final['frames']:
        original_pins[frame['float32_arrays']] = sha(frame['float32_arrays'])
    runtime_sources = {str(Path(binding['repository']) / relative): value for relative, value in binding['replica_source_sha256'].items()}
    for relative, expected in binding['original_source_sha256'].items():
        original_pins[str(Path(binding['original_repository']) / relative)] = expected
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU variant preparation initialized CUDA')
    plan.update(status='single_variable_opacity_outside_prepared_not_started', repository=binding['repository'], training_config=current, training_config_changed_fields=['model_path', 'start_checkpoint'], training_config_added_fields=binding['added_training_fields'], original_file_sha256=original_pins, runtime_source_sha256=runtime_sources, experiment_binding_path=str(Path(binding_path).resolve()), experiment_binding_sha256=sha(binding_path), guard_path=binding['guard_path'], guard_sha256=binding['guard_sha256'], runtime_helper_import_contract=binding['runtime_helper_import_contract'], first_step_reference=binding['first_step_pair_reference'], all_iteration_pair_reference=binding['all_iteration_pair_reference'], baseline_terminal_evaluation=str(final_eval_path), preparation_checks=checks, all467_original_inputs_hash_verified=True, independent_actual_Scene_per_frame_pose_receipt_available=False, training_math='Full frozen original forward/loss/backward/Adam with only bound .05 outside-opacity hypothesis addition. All10 original first-step scalars checked exactly BEFORE first backward. Original paper terms/SGCM/reset/horizon/seed preserved.', algorithm_changes=binding['algorithm_changed_files'], scientific_status=binding['scientific_status'], allow_training=False, first_backward_policy='Intercept only real training.training first Tensor.backward; compare original10 scalars using pre-add original loss capture, require exact equality and actual restore pass before invoking original backward.', paired_trajectory_policy='Require exact iteration/camera/time_shift/LR equality for every one of300 post-step records; preserve every extra-eval RNG.', paired_evaluation_policy='At each node require exact baseline49 GT float32 arrays and matching frame IDs/dimensions/each region pixel count; evaluation math remains baseline code.', GPU_ownership='CPU preparation only by default. Root exact variant receipt plus independent final review and GPU handoff needed; same exclusive flock and idle check.', launcher_path=str(PLAN / 'launch_opacity_outside_continuation.py'), launcher_sha256=sha(PLAN / 'launch_opacity_outside_continuation.py'))
    return plan, cfg, state


def verify_paired_evaluation(result, baseline, np):
    lookup = {frame['frame_id']: frame for frame in baseline['frames']}
    rows = []
    for frame in result['frames']:
        reference = lookup[frame['frame_id']]
        with np.load(frame['float32_arrays']) as actual_npz, np.load(reference['float32_arrays']) as saved_npz:
            exact_gt = np.array_equal(actual_npz['gt'], saved_npz['gt'])
            shape = list(actual_npz['gt'].shape)
        counts = {key: item['pixel_count'] for key, item in frame['regions'].items()}
        expected_counts = {key: item['pixel_count'] for key, item in reference['regions'].items()}
        if not exact_gt or shape != [3, 320, 480] or counts != expected_counts:
            raise RuntimeError('Actual paired evaluation GT/dimensions/mask domain differs: ' + frame['frame_id'])
        rows.append({'frame_id': frame['frame_id'], 'GT_float32_exact_equal': True, 'GT_shape': shape, 'region_pixel_counts_exact_equal': True, 'region_pixel_counts': counts})
    return {'status': 'actual_paired_inputs_and_mask_domains_exact', 'iteration': result['iteration'], 'frame_count': len(rows), 'rows': rows, 'original_loader_camera_transform_and_all467_data_SHA_verified': True, 'independent_actual_Scene_pose_receipt': False}


class DiagnosticComplete(Exception):
    pass


def run(plan, cfg, sidecar, approval_path):
    approval = json.loads(approval_path.read_text())
    if approval.get('decision') != 'APPROVED_SINGLE_VARIABLE_OPACITY_OUTSIDE_30K_TO30300' or approval.get('prepared_plan_sha256') != sha(VARIANT_PLAN) or approval.get('harness_sha256') != plan['harness_sha256'] or approval.get('observer_sha256') != plan['observer_sha256'] or approval.get('experiment_binding_sha256') != plan['experiment_binding_sha256'] or approval.get('guard_sha256') != plan['guard_sha256'] or approval.get('first_iteration') != FIRST or approval.get('stop_iteration') != STOP:
        raise RuntimeError('Root receipt does not bind exact single-variable hypothesis plan')
    review_path = Path(approval['independent_review_path'])
    if sha(review_path) != approval['independent_review_sha256']:
        raise RuntimeError('Independent final review SHA differs from root receipt')
    review = json.loads(review_path.read_text())
    if review.get('status') != 'PASS' or review.get('harness_sha256') != plan['harness_sha256'] or review.get('prepared_plan_sha256') != sha(VARIANT_PLAN) or review.get('experiment_binding_sha256') != plan['experiment_binding_sha256']:
        raise RuntimeError('Independent final review does not bind this exact runnable plan')
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise RuntimeError('Set PYTHONDONTWRITEBYTECODE=1 to keep frozen source read-only')
    if os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '') != plan['allocator_environment_original']:
        raise RuntimeError('Allocator environment differs from saved formal child')
    lock = (BASE / 'campaign/gpu_slot.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError('GPU compute must be idle before controlled continuation')
    import uuid
    owner = {'owner_pid': os.getpid(), 'nonce': uuid.uuid4().hex, 'purpose': 'Root-approved isolated30k->30300 single outside-opacity hypothesis', 'output': cfg.model_path, 'started_utc': now()}
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps(owner))
    lock.flush()
    os.fsync(lock.fileno())
    output = Path(cfg.model_path)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'diagnostic_plan.json', plan)
    write(output / 'root_coordination_receipt.json', approval)
    status = dict(owner, status='initializing', parent_iteration=30000, stop_iteration=STOP, gpu_lock_held=True)
    write(output / 'status.json', status)
    exit_code = 1
    try:
        REPO = Path(plan['repository']).resolve()
        binding, hypothesis_guard = load_experiment(plan['experiment_binding_path'])
        hypothesis_guard.verify_sources()
        hypothesis_guard.verify_protected_files()
        hypothesis_guard.verify_auxiliary_files()
        os.chdir(REPO)
        sys.path.insert(0, str(PLAN))
        sys.path.insert(0, str(REPO))
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from torchvision.utils import save_image
        if torch.__version__ != '2.2.2+cu118' or torch.version.cuda != '11.8' or torch.is_autocast_enabled():
            raise RuntimeError('Original torch2.2.2cu118/no-AMP runtime required')
        spec = importlib.util.spec_from_file_location('unchanged_formal_training', REPO / 'train.py')
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        import gaussian_renderer
        import utils.paper_checkpoint as guard
        import scene.waymo_loader2 as loader
        import scene.gaussian_model as gaussian_model
        import scene.cameras as camera_module
        import scene.envlight as env_module
        import utils.paper_losses as paper_losses
        import utils.paper_masks as paper_masks
        import utils.loss_utils as loss_utils
        import utils.general_utils as general_utils
        import utils.sh_utils as sh_utils
        import scene.paper_sgcm as sgcm_module
        import diff_gaussian_rasterization as rasterizer
        runtime_import_pins = {}
        for module, expected in [(training, REPO / 'train.py'), (gaussian_renderer, REPO / 'gaussian_renderer/__init__.py'), (gaussian_model, REPO / 'scene/gaussian_model.py'), (camera_module, REPO / 'scene/cameras.py'), (env_module, REPO / 'scene/envlight.py'), (guard, REPO / 'utils/paper_checkpoint.py'), (loader, REPO / 'scene/waymo_loader2.py'), (paper_losses, REPO / 'utils/paper_losses.py'), (paper_masks, REPO / 'utils/paper_masks.py'), (loss_utils, REPO / 'utils/loss_utils.py'), (general_utils, REPO / 'utils/general_utils.py'), (sh_utils, REPO / 'utils/sh_utils.py'), (sgcm_module, REPO / 'scene/paper_sgcm.py')]:
            actual_path = Path(module.__file__).resolve()
            if actual_path != expected or sha(actual_path) != plan['runtime_source_sha256'][str(expected)]:
                raise RuntimeError('Imported formal training dependency differs from actual pin: ' + str(actual_path))
            runtime_import_pins[str(actual_path)] = sha(actual_path)
        if training.render is not gaussian_renderer.render:
            raise RuntimeError('Training bound a different renderer function')
        if training.ssim is not loss_utils.ssim or training.psnr is not loss_utils.psnr:
            raise RuntimeError('Training bound different checked SSIM/PSNR functions')
        for actual in (Path(rasterizer.__file__).resolve(), Path(rasterizer._C.__file__).resolve()):
            if str(actual) not in plan['original_file_sha256'] or sha(actual) != plan['original_file_sha256'][str(actual)]:
                raise RuntimeError('Imported rasterizer Python/binary differs from pinned original: ' + str(actual))
            runtime_import_pins[str(actual)] = sha(actual)
        for name, row in plan['runtime_helper_import_contract'].items():
            module = importlib.import_module(name) if name == 'runtime_safety_parameterized' else None
            if module is not None and (Path(module.__file__).resolve() != Path(row['path']).resolve() or sha(module.__file__) != row['sha256']):
                raise RuntimeError('Actual restore observer import differs from reviewed helper pin')
            if module is not None:
                runtime_import_pins[str(Path(module.__file__).resolve())] = sha(module.__file__)
        write(output / 'runtime_import_pins.json', {'status': 'actual_import_paths_and_sha_verified', 'pins': runtime_import_pins})
        from runtime_safety_parameterized import ReadOnlyLoaderNumpy, install_restore_observer, optimizer_receipt
        # Diagnostic helper functions are observations only; do not import/run its main.
        sys.path.insert(0, str(PLAN.parent / 'scripts'))
        from component_diagnostic import stats, rgba_package, region_stats, rng_capture, rng_restore, rng_pins
        import component_diagnostic as diagnostic_helper
        helper_pin = plan['runtime_helper_import_contract']['component_diagnostic']
        if Path(diagnostic_helper.__file__).resolve() != Path(helper_pin['path']).resolve() or sha(diagnostic_helper.__file__) != helper_pin['sha256']:
            raise RuntimeError('Actual metric/camera helper import path/SHA differs from reviewed baseline')
        runtime_import_pins[str(Path(diagnostic_helper.__file__).resolve())] = sha(diagnostic_helper.__file__)
        runtime_import_pins[plan['guard_path']] = sha(plan['guard_path'])
        write(output / 'runtime_import_pins.json', {'status': 'actual_replica_and_unchanged_helper_import_paths_SHA_verified', 'pins': runtime_import_pins})
        original_config_validator = hypothesis_guard.install_on_isolated_module(guard)
        guard._validate_resume_configuration(sidecar['config'], cfg)
        loader_numpy_original = loader.np
        readonly_np = ReadOnlyLoaderNumpy(loader.np, cfg.source_path)
        loader.np = readonly_np
        restore = {}
        original_restore = install_restore_observer(guard, sidecar, restore, torch, np, expected_iteration=30000)
        torch.cuda.set_device(torch.device('cuda:0'))
        training.seed_everything(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        original_eval = training.complete_eval
        loss_stream = (output / 'step_losses.jsonl').open('w')
        node_records = []
        paired_baseline_eval = json.loads(Path(plan['baseline_terminal_evaluation']).read_text())
        paired_baseline_rows = {row['iteration']: row for row in plan['all_iteration_pair_reference']['rows']}
        original_tensor_backward = torch.Tensor.backward
        first_backward_receipt = {}

        def before_first_backward(tensor, *positional, **keyword):
            caller = inspect.currentframe().f_back
            is_first_training_backward = caller.f_code is training.training.__code__ and int(caller.f_locals['iteration']) == FIRST
            if is_first_training_backward:
                if tensor is not caller.f_locals['loss'] or first_backward_receipt:
                    raise RuntimeError('Unexpected/repeated real first training backward')
                if restore.get('status') != 'actual_restore_pass':
                    raise RuntimeError('Actual saved30k restoration must pass before first backward')
                log = caller.f_locals['log_dict']
                original_loss = float(log['loss_original_objective_before_diagnostic'])
                with torch.no_grad():
                    first_psnr = float(training.psnr(caller.f_locals['image'], caller.f_locals['gt_image']).double().item())
                actual = {key: original_loss if key == 'loss' else first_psnr if key == 'psnr' else float(log[key].item()) if torch.is_tensor(log[key]) else float(log[key]) for key in plan['first_step_reference']['values']}
                comparisons = {key: {'actual': value, 'reference': plan['first_step_reference']['values'][key], 'exact_equal': value == plan['first_step_reference']['values'][key], 'abs_delta': abs(value - plan['first_step_reference']['values'][key])} for key, value in actual.items()}
                first_backward_receipt.update(iteration=FIRST, phase='before_actual_first_Tensor_backward', camera_id=guard.camera_id(caller.f_locals['viewpoint_cam']), original_values=actual, comparisons=comparisons, actual_new_total_loss=float(tensor.item()), original_objective_loss=original_loss, background_opacity_unweighted=float(log['loss_diagnostic_dynamic_background_opacity']), added_weight=float(cfg.diagnostic_lambda_dynamic_background_opacity), tolerance_applied=False, original_backward_invoked=False)
                write(output / 'first_backward_reference_comparison.json', first_backward_receipt)
                if not all(row['exact_equal'] for row in comparisons.values()) or first_backward_receipt['camera_id'] != restore['next_camera_id'] or caller.f_locals['viewpoint_stack'] != restore['next_remaining_stack']:
                    raise RuntimeError('Original first10 scalars/camera are not exactly paired; backward not invoked')
            result = original_tensor_backward(tensor, *positional, **keyword)
            if is_first_training_backward:
                first_backward_receipt['original_backward_invoked'] = True
                first_backward_receipt['original_backward_returned'] = True
                write(output / 'first_backward_reference_comparison.json', first_backward_receipt)
            return result

        torch.Tensor.backward = before_first_backward

        @torch.no_grad()
        def diagnostic_eval(iteration, scene, models, background):
            rng_before = rng_capture()
            result = {'iteration': iteration, 'RNG_before': rng_pins(rng_before), 'frames': [], 'model_statistics': {}}
            try:
                for branch in ('static', 'dynamic'):
                    opacity = models[branch].get_opacity
                    result['model_statistics'][branch] = {'point_count': len(models[branch].get_xyz), 'opacity': stats(opacity), 'opacity_fraction_le_0_01': float((opacity <= .01).float().mean())}
                result['model_statistics']['environment'] = {'global': stats(models['environment'].base), 'faces': [stats(models['environment'].base[i]) for i in range(6)]}
                lookup = {guard.camera_id(cam): cam for cam in scene.getTrainCameras()}
                from PIL import Image
                import cv2
                selected_ids = sorted(lookup) if iteration == STOP else FRAME_IDS
                if iteration == STOP and len(selected_ids) != 49:
                    raise RuntimeError('Terminal evaluation requires all49 original cameras')
                for frame_id in selected_ids:
                    camera = lookup[frame_id]
                    package = gaussian_renderer.render(camera, models['static'], models['dynamic'], cfg, background, env_map=models['environment'], is_training=False)
                    env = models['environment'](camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
                    values = rgba_package(package, env, camera.original_image)
                    actor_path = Path(cfg.source_path) / 'dynamic_masks/dynamic_mask_0' / (camera.image_name + '.png')
                    actor = np.asarray(Image.open(actor_path).convert('L').resize((camera.image_width, camera.image_height), Image.Resampling.NEAREST)) > 0
                    sky = camera.sky_mask.detach().cpu().numpy()[0] > 0
                    masks = {'full': np.ones_like(actor), 'sky': sky, 'non_sky': ~sky, 'actor': actor, 'outside_actor': ~actor, 'non_sky_outside_actor': ~sky & ~actor}
                    paths = {}
                    for channel, data in [('gt', camera.original_image), ('composite', package['render']), ('static', package['render_s']), ('dynamic', package['render_d']), ('alpha_s', package['static_alpha']), ('alpha_d', package['dynamic_alpha'])]:
                        path = output / 'extra_eval' / str(iteration) / channel / (camera.image_name + '.png')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        save_image(data.clamp(0, 1), path)
                        paths[channel] = str(path)
                    float_path = output / 'extra_eval' / str(iteration) / 'float32' / (camera.image_name + '.npz')
                    float_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(float_path, **values)
                    result['frames'].append({'frame_id': frame_id, 'outputs': paths, 'float32_arrays': str(float_path), 'regions': region_stats(values, masks, values['gt'])})
                    del package, env, values
            finally:
                rng_restore(rng_before)
            result['RNG_after_restoration'] = rng_pins(rng_capture())
            result['all_rng_unchanged'] = result['RNG_after_restoration'] == result['RNG_before']
            if not result['all_rng_unchanged']:
                raise RuntimeError('Extra eval polluted training RNG trajectory')
            write(output / 'extra_eval' / str(iteration) / 'diagnostic_eval.json', result)
            return result

        def tracked_eval(*positional, **keyword):
            caller = inspect.currentframe().f_back
            iteration = int(positional[1])
            losses = {key: float(value.item()) if torch.is_tensor(value) else float(value) for key, value in positional[6].items()}
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError('Nonfinite original full objective diagnostics at ' + str(iteration))
            if restore.get('status') != 'actual_restore_pass' or not readonly_np.checked:
                raise RuntimeError('Exact restoration and read-only input normalization must pass')
            camera_id = guard.camera_id(caller.f_locals['viewpoint_cam'])
            if iteration == FIRST and (camera_id != restore['next_camera_id'] or caller.f_locals['viewpoint_stack'] != restore['next_remaining_stack']):
                raise RuntimeError('First real camera differs from saved sampler/RNG')
            models = {'static': caller.f_locals['gaussians'], 'dynamic': caller.f_locals['dynamic_gaussians'], 'environment': caller.f_locals['env_map']}
            shift = caller.f_locals.get('time_shift')
            if torch.is_tensor(shift) or isinstance(shift, np.generic):
                shift = float(shift)
            record = {'iteration': iteration, 'camera_id': camera_id, 'time_shift': shift, 'losses': losses, 'ema_after': {k: float(v) for k, v in caller.f_locals['ema_dict_for_log'].items()}, 'learning_rates': {name: {group.get('name', 'base'): float(group['lr']) for group in model.optimizer.param_groups} for name, model in models.items()}}
            paired_actual = {key: record[key] for key in ('iteration', 'camera_id', 'time_shift', 'learning_rates')}
            if paired_actual != paired_baseline_rows[iteration]:
                write(output / 'paired_trajectory_mismatch.json', {'actual': paired_actual, 'reference': paired_baseline_rows[iteration]})
                raise RuntimeError('Actual300-step camera/time_shift/LR trajectory differs from original baseline')
            record['paired_original_camera_time_shift_LR_exact'] = True
            if iteration == FIRST:
                if not first_backward_receipt.get('original_backward_invoked') or not all(row['exact_equal'] for row in first_backward_receipt['comparisons'].values()):
                    raise RuntimeError('Strict first10 before-backward comparison was not completed')
                comparisons = {}
                for key, reference in plan['first_step_reference']['values'].items():
                    if key not in losses:
                        raise RuntimeError('First original-math step lacks reference diagnostic: ' + key)
                    actual = losses['loss_original_objective_before_diagnostic'] if key == 'loss' else losses[key]
                    comparisons[key] = {'actual': actual, 'reference': reference, 'exact_equal': actual == reference, 'abs_delta': abs(actual - reference)}
                record['first_step_existing_tensorboard_comparison'] = comparisons
                write(output / 'first_step_reference_comparison.json', {'iteration': FIRST, 'camera_id': camera_id, 'reference_pin': plan['first_step_reference'], 'actual_values': losses, 'comparisons_original_loss_before_added_term': comparisons, 'first_backward_reference': first_backward_receipt, 'tolerance_applied': False, 'comparison_does_not_claim_cuda_bitwise_determinism': True})
            loss_stream.write(json.dumps(record, allow_nan=False) + '\n')
            loss_stream.flush()
            if iteration in NODES:
                for name, model in models.items():
                    if any(not torch.isfinite(p).all() for group in model.optimizer.param_groups for p in group['params']) or any(not torch.isfinite(v).all() for state in model.optimizer.state.values() for v in state.values() if torch.is_tensor(v)):
                        raise FloatingPointError('Nonfinite post-step parameter/Adam: ' + name)
                record['optimizer_after_step'] = optimizer_receipt(models)
            result = original_eval(*positional, **keyword)
            if iteration in NODES:
                # Release detached caches only; no model tensor/math/optimizer mutations.
                evaluated = diagnostic_eval(iteration, positional[3], models, positional[5][1])
                paired_inputs = verify_paired_evaluation(evaluated, paired_baseline_eval, np)
                write(output / 'extra_eval' / str(iteration) / 'paired_input_domains.json', paired_inputs)
                node_records.append(dict(record, diagnostic_eval=evaluated, paired_input_domains=paired_inputs))
                status.update(status='controlled_training', iteration=iteration, elapsed_seconds=time.monotonic() - started, last_loss=losses, restore=restore, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                write(output / 'status.json', status)
            if iteration >= STOP:
                terminal = output / 'terminal_30300'
                terminal.mkdir(exist_ok=False)
                components = guard.save_components(terminal, iteration, models)
                continuation = guard.capture_training_state(iteration, positional[3], caller.f_locals['viewpoint_stack'], models, cfg, sgcm=caller.f_locals['paper_sgcm'], ema=caller.f_locals['ema_dict_for_log'])
                state_path = terminal / ('training_state' + str(iteration) + '.pth')
                torch.save(continuation, state_path)
                components['training_state'] = {'path': str(state_path), 'bytes': state_path.stat().st_size}
                for branch, item in components.items():
                    item['sha256'] = sha(item['path'])
                    saved = torch.load(item['path'], map_location='cpu', weights_only=False)
                    if branch == 'training_state':
                        if saved['iteration'] != iteration or saved['config'] != plan['training_config'] or saved['viewpoint_stack'] != continuation['viewpoint_stack']:
                            raise RuntimeError('Saved terminal continuation iteration/config/sampler mismatch')
                    else:
                        capture, saved_iteration = saved
                        if saved_iteration != iteration or len(capture) != (2 if branch == 'environment' else 18):
                            raise RuntimeError('Saved terminal component iteration/schema mismatch')
                    item['cpu_load_iteration_schema_validated'] = True
                    del saved
                snapshot = {'schema_version': 1, 'artifact_kind': 'isolated_controlled_continuation_diagnostic_snapshot', 'status': 'complete', 'iteration': iteration, 'parent_iteration': 30000, 'first_training_iteration': FIRST, 'horizon_retained': 50000, 'seed': 0, 'checkpoints': components, 'training_config': plan['training_config'], 'source_pins': runtime_import_pins, 'terminal_evaluation': str(output / 'extra_eval' / str(iteration) / 'diagnostic_eval.json'), 'terminal_evaluation_camera_count': 49, 'continuation_RNG_restored_after_extra_eval': True, 'not_a_formal_paper_checkpoint_event': True, 'formal_run_modified': False}
                write(terminal / 'diagnostic_snapshot_manifest.json', snapshot)
                torch.cuda.synchronize()
                raise DiagnosticComplete()
            return result

        training.complete_eval = tracked_eval
        OmegaConf.save(cfg, output / 'effective_config.yaml')
        try:
            training.training(cfg)
            raise RuntimeError('Observer failed to stop controlled original50k loop')
        except DiagnosticComplete:
            if (output / 'checkpoint_events').exists():
                raise RuntimeError('Controlled diagnostic published an unexpected formal event')
            report = {'status': 'controlled_30k_to30300_single_variable_opacity_outside_hypothesis_complete', 'first_iteration': FIRST, 'stop_iteration': STOP, 'horizon': 50000, 'seed': 0, 'step_count': STOP - FIRST + 1, 'restore': restore, 'nodes': node_records, 'runtime_import_pins': runtime_import_pins, 'terminal_snapshot_manifest': str(output / 'terminal_30300/diagnostic_snapshot_manifest.json'), 'terminal_full49_evaluation': True, 'all_node_eval_rng_preserved': all(row['diagnostic_eval']['all_rng_unchanged'] for row in node_records), 'all300_camera_time_shift_LR_exactly_paired': True, 'all49_GT_float32_and_region_pixel_counts_exactly_paired': True, 'first10_original_scalars_exact_before_backward': first_backward_receipt, 'genuine_full_loss_backward_Adam': True, 'original_algorithm_changed': True, 'single_algorithm_change': plan['scientific_status'], 'formal_original_source_changed': False, 'formal50k_resumed': False, 'no_formal_checkpoint_event': True, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'elapsed_seconds': time.monotonic() - started}
            write(output / 'continuation_diagnostic.json', report)
            for path, expected in plan['original_file_sha256'].items():
                if sha(path) != expected:
                    raise RuntimeError('Original protected file changed during diagnostic: ' + path)
            hypothesis_guard.verify_sources()
            hypothesis_guard.verify_auxiliary_files()
            hypothesis_guard.verify_protected_files()
            if (Path(cfg.source_path) / 'dynamic_masks/shadow_mask_0').exists():
                raise RuntimeError('Unbound new appearance shadow mask appeared during diagnostic')
            status.update(status='complete', no_modification_sha_pass=True, actual_exit_code=0)
            exit_code = 0
        finally:
            torch.Tensor.backward = original_tensor_backward
            training.complete_eval = original_eval
            guard.restore_training_state = original_restore
            guard._validate_resume_configuration = original_config_validator
            loader.np = loader_numpy_original
            loss_stream.close()
    except BaseException as error:
        status.update(status='failed', actual_exit_code=130 if isinstance(error, KeyboardInterrupt) else 1, error=repr(error), traceback=traceback.format_exc())
        exit_code = status['actual_exit_code']
    finally:
        status.update(gpu_lock_held=False, finished_utc=now())
        write(output / 'status.json', status)
        (output / 'actual_exit_code.txt').write_text(str(exit_code) + '\n')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-binding', type=Path, required=True)
    parser.add_argument('--root-approved-plan', type=Path)
    args = parser.parse_args()
    binding = json.loads(args.experiment_binding.read_text())
    plan, cfg, state = prepare(Path(binding['output']), args.experiment_binding)
    if args.root_approved_plan is None:
        write(VARIANT_PLAN, plan)
        print(json.dumps({'status': plan['status'], 'training_started': False, 'CUDA_initialized': False, 'allow_training': False, 'prepared_plan': str(VARIANT_PLAN), 'prepared_plan_sha256': sha(VARIANT_PLAN), 'harness_sha256': plan['harness_sha256'], 'binding_sha256': plan['experiment_binding_sha256']}))
        return 0
    saved = json.loads(VARIANT_PLAN.read_text())
    if saved != plan:
        raise RuntimeError('Prepared plan no longer matches actual code/config/source pins')
    return run(plan, cfg, state, args.root_approved_plan)


if __name__ == '__main__':
    sys.exit(main())
