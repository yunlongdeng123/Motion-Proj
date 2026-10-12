"""Focused synthetic gate/idempotency tests; no training or CUDA imports."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PATH = Path(__file__).with_name('prepare_long_variant_binding_cpu.py')
spec = importlib.util.spec_from_file_location('long_binding_CPU_under_test', PATH)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
results = []


def check(name, action, reject=False):
    try:
        action()
    except (ValueError, m.Incomplete):
        if not reject:
            raise
    else:
        if reject:
            raise AssertionError('Unexpected accept: ' + name)
    results.append({'check': name, 'pass': True})


def main():
    import numpy as np
    plan = {'parent_iteration': 30000, 'first_iteration': 30001, 'stop_iteration': 32000, 'steps': 2000, 'horizon': 50000, 'seed': 0, 'algorithm_changes': [], 'repository': str(m.ORIGINAL), 'extra_eval_nodes': [30001, 30300, 31000, 32000], 'terminal_eval_camera_count': 49}
    check('actual_long_window_contract_accepts', lambda: m.window(plan))
    for key, value in (('stop_iteration', 30300), ('steps', 300), ('horizon', 32000), ('seed', False), ('algorithm_changes', ['train.py']), ('extra_eval_nodes', [30001, 32000])):
        changed = dict(plan); changed[key] = value
        check('reject_' + key, lambda changed=changed: m.window(changed), reject=True)
    work = PATH.parents[3]
    for arm, output_name in (('B1', 'original30k_to32000'), ('B2', 'original30k_to32000_noise_reference')):
        release = json.loads((work / 'continuation_diagnostic' / output_name / 'receipts/gpu_release_handoff.json').read_bytes())
        check('actual_' + arm + '_release_schema_accepts', lambda arm=arm, release=release: m.validate_release(arm, release))
        exit_key = 'actual_worker_exit_code' if arm == 'B1' else 'worker_exit_code'
        reject = dict(release); reject[exit_key] = 1
        check(arm + '_nonzero_actual_worker_exit_rejected', lambda arm=arm, reject=reject: m.validate_release(arm, reject), reject=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        generated = root / 'only_generated.json'
        check('first_exclusive_generated_write', lambda: m.require(m.write_once(generated, b'{"ok":1}'), 'not created'))
        before = generated.stat().st_mtime_ns
        check('idempotent_identical_file_is_not_rewritten', lambda: m.require(m.write_once(generated, b'{"ok":1}') is False and generated.stat().st_mtime_ns == before, 'rewrote identical output'))
        check('conflicting_prior_generated_file_rejected', lambda: m.write_once(generated, b'{"ok":2}'), reject=True)
        check('conflicting_file_preserved', lambda: m.require(generated.read_bytes() == b'{"ok":1}', 'overwritten'))
        gt = np.zeros((3, 320, 480), dtype=np.float32)
        good = root / 'true_shape.npz'; np.savez_compressed(good, gt=gt)
        bad = root / 'wrong_dtype.npz'; np.savez_compressed(bad, gt=gt.astype(np.float64))
        source = {'plan': {'original_file_sha256': {'initial4': 'exact'}, 'training_config': {'model_path': 'B1', 'iterations': 50000}, 'first_step_reference': {'values': {f'q{i}': 0 for i in range(10)}}}, 'rows': [{'iteration': i, 'camera_id': f'c{i%49}', 'time_shift': 0.0, 'learning_rates': {'xyz': .001}, 'losses': {f'q{k}': 0.1 for k in range(10)}} for i in range(30001, 32001)], 'final': {'frames': [{'frame_id': f, 'regions': {'full': {'pixel_count': 153600}, 'actor': {'pixel_count': 20}}, 'float32_arrays': str(good)} for f in m.FRAMES]}}
        repeat = copy.deepcopy(source); repeat['plan']['training_config']['model_path'] = 'B2'
        check('all2000_sampling_and_all49_true_array_pair_accepts', lambda: m.require(len(m.paired(source, repeat)[0]) == 2000 and len(m.paired(source, repeat)[2]) == 49, 'missing pair rows'))
        for name, alter in (
            ('sample_mismatch_at_late1999', lambda x: x['rows'][1999].update(camera_id='different')),
            ('time_shift_mismatch', lambda x: x['rows'][1200].update(time_shift=0.1)),
            ('LR_mismatch', lambda x: x['rows'][1500].update(learning_rates={'xyz': .002})),
            ('initial4_source_kernel_pin_mismatch', lambda x: x['plan']['original_file_sha256'].update(initial4='different')),
            ('other_original_config_field_mismatch', lambda x: x['plan']['training_config'].update(iterations=32000)),
            ('first10_original_scalar_mismatch', lambda x: x['rows'][0]['losses'].update(q0=.2)),
            ('region_count_mismatch_even_same_image', lambda x: x['final']['frames'][0]['regions']['actor'].update(pixel_count=21)),
            ('GT_float64_not_accepted', lambda x: x['final']['frames'][0].update(float32_arrays=str(bad))),
        ):
            different = copy.deepcopy(repeat); alter(different)
            check(name, lambda different=different: m.paired(source, different), reject=True)
        wait_root = root / 'waiting'; wait_root.mkdir()
        canonical = wait_root / 'v_binding_cpu'; canonical.mkdir()
        fake_script = canonical / PATH.name; fake_script.write_bytes(PATH.read_bytes())
        with patch.object(m, 'ROOT', wait_root), patch.object(m, 'HERE', canonical), patch.object(m, '__file__', str(fake_script)), patch.object(m, 'PINNED', {}), patch.object(m, 'completed', side_effect=AssertionError('Must not scan completed artifacts while B2 is incomplete')), patch.object(m, 'materialize', side_effect=AssertionError('Must not create V while B2 incomplete')), patch.object(sys, 'argv', ['test', '--materialize-and-cpu-prepare']), contextlib.redirect_stdout(io.StringIO()):
            m.main()
        gate = json.loads((canonical / 'gate_status.json').read_bytes())
        check('even_materialize_flag_waits_without_real_B2_release', lambda: m.require(gate['status'] == 'WAIT_REAL_B2_COMPLETION_AND_EXPORT' and gate['V_binding_created_by_this_call'] is False and gate['V_replica_created_by_this_call'] is False and gate['V_CPUplan_created_by_this_call'] is False, 'premature generation'))
        check('only_own_gate_status_created_while_waiting', lambda: m.require(sorted(p.name for p in canonical.iterdir()) == ['gate_status.json', PATH.name], 'unexpected V files'))
    tree = ast.parse(PATH.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in imports if isinstance(n, ast.Import) for a in n.names] + [n.module for n in imports if isinstance(n, ast.ImportFrom)]
    check('generator_has_no_torch_renderer_training_imports', lambda: m.require(not any(n and (n == 'torch' or n.startswith(('gaussian_renderer', 'scene.', 'train'))) for n in names), 'GPU import'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'subprocess']
    check('only_subprocess_is_no_approval_CPU_prepare', lambda: m.require(len(calls) == 1 and calls[0].func.attr == 'run' and '--root-approved-plan' not in PATH.read_text() and 'nvidia-smi' not in PATH.read_text(), 'launch path exists'))
    output = {'status': 'PASS', 'focused_check_count': len(results), 'checks': results, 'CPU_only': True, 'synthetic_tests_only_not_actual_B2_completion': True, 'actual_V_CUDA_restore': False, 'generator_sha256': m.sha(PATH)}
    Path(__file__).with_name('focused_generator_tests_cpu.json').write_bytes(m.payload(output))
    print(json.dumps({'status': output['status'], 'check_count': len(results), 'generator_sha256': output['generator_sha256']}))


if __name__ == '__main__':
    main()
