"""One blocking-lock terminal collection; no training polling or GPU/model calls."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013')
RUN = BASE / 'runs/scene0029_069_117_hybrid_30000_seed0_GN11'
TOOLS = BASE / 'tools/red_terminal_20261013'
META = BASE / 'metadata/red_terminal_20261013_1056_attempt2'
REPORT = BASE / 'reports/red_gn11_30k_saved_cpu_20261013'
PY = '/root/autodl-tmp/envs/drivingeditor/bin/python'
OWNER_PID = 124555
STATE = {'schema_version': 1, 'collector_pid': os.getpid(), 'owned_training_pid': OWNER_PID,
         'run': str(RUN), 'report': str(REPORT), 'metadata': str(META),
         'collector_log': str(META / 'collector.stdout.log'), 'stages': [],
         'GPU_used': False, 'Torch_imported': False, 'training_started': False,
         'continuous_training_polling': False, 'restoration_verified': False,
         'new_hash_or_fingerprint_computed': False, 'human_verdict': None,
         'prior_tooling_attempt_preserved': str(BASE / 'metadata/red_terminal_20261013_1056')}
ACTIVE = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def save(status, **extra):
    STATE.update(status=status, updated_utc=now(), **extra)
    temporary = META / 'status.json.tmp'
    temporary.write_text(json.dumps(STATE, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(META / 'status.json')
    with (META / 'lifecycle.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'status': status, 'collector_pid': os.getpid(), 'updated_utc': STATE['updated_utc']}) + '\n')
    print(json.dumps({'status': status, 'collector_pid': os.getpid(), 'updated_utc': STATE['updated_utc']}), flush=True)


def wait_owned_exit(pid, timeout=30):
    """Existing bounded /proc compatibility path, only after verified terminal state."""
    proc = Path('/proc') / str(pid)
    deadline = time.monotonic() + timeout
    while True:
        try:
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            command = (proc / 'cmdline').read_bytes().decode('utf-8', errors='replace')
        except FileNotFoundError:
            return {'method': 'bounded_proc_exit_wait', 'pidfd_available': hasattr(os, 'pidfd_open'), 'owner_already_exited': True}
        if state in ('Z', 'X'):
            return {'method': 'bounded_proc_exit_wait', 'pidfd_available': hasattr(os, 'pidfd_open'), 'owner_exit_confirmed': True}
        if command:
            assert 'run_scene_gn11.py' in command and 'scene0029_spec.json' in command, 'PID identity changed; preserve unrelated process'
        assert time.monotonic() < deadline, 'Owned red PID has not exited; stop without saved arrays'
        time.sleep(.25)


def stage(name, command, json_result=None):
    global ACTIVE
    entry = {'name': name, 'command': command, 'started_utc': now()}
    STATE['stages'].append(entry)
    with (META / (name + '.stdout.log')).open('xb') as output:
        ACTIVE = subprocess.Popen(command, cwd=str(BASE), stdin=subprocess.DEVNULL,
                                  stdout=output, stderr=subprocess.STDOUT, close_fds=True,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                                           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4'))
        entry['pid'] = ACTIVE.pid
        save(name + '_running', active_cpu_child_pid=ACTIVE.pid)
        entry['returncode'] = ACTIVE.wait()
        ACTIVE = None
    entry['completed_utc'] = now()
    assert entry['returncode'] == 0, name + ' failed; preserve actual partial files'
    if json_result is not None:
        result = read(META / (name + '.stdout.log'))
        write_new(json_result, result)
    save(name + '_complete', active_cpu_child_pid=None)


def copy_new(source, destination):
    source, destination = Path(source), Path(destination)
    assert source.is_file() and not destination.exists()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    shutil.copystat(source, destination)


def stopped(signum, frame):
    raise KeyboardInterrupt('Owned CPU collector received signal ' + str(signum))


def main():
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        assert META.is_dir() and not REPORT.exists() and not REPORT.with_suffix('.tar.gz').exists()
        save('waiting_shared_GPU_lock_for_owned_red_terminal')
        with Path('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            save('shared_lock_acquired_read_terminal_once')
            status, memory = read(RUN / 'status.json'), read(RUN / 'memory_summary.json')
            assert status['status'] == 'complete' and status['iteration'] == 30000 and status['pid'] == OWNER_PID, 'Not owned red 30k terminal; stop'
            assert memory['monitor_status'] == 'closed' and memory['run_status'] == 'complete' and memory['iteration'] == 30000, 'Terminal memory not closed; stop'
            assert memory['error'] is None and memory['error_count'] == 0, 'Terminal resource error; stop'
            assert status.get('error') is None, 'Terminal status error; stop'
            STATE['owned_training_exit_check'] = wait_owned_exit(OWNER_PID)
            scene_binding, guard_binding = read(RUN / 'actual_scene_binding.json'), read(RUN / 'actual_sgcm_binding.json')
            assert scene_binding['status'] == 'PASS_ACTUAL_GPU_SCENE_BINDING' and scene_binding['scene_id'] == 'scene0029_069_117'
            assert guard_binding['status'] == 'PASS_ACTUAL_GPU_SGCM_LINEAGE_BINDING' and guard_binding['guard_enabled'] is True and guard_binding['training_views'] == 49
            RAW = META / 'original_snapshot'
            RAW.mkdir()
            for name in ('status.json', 'memory_summary.json', 'memory_samples.jsonl', 'memory_phases.jsonl',
                         'gradient_diagnostics.jsonl', 'effective_config.json', 'actual_scene_binding.json', 'actual_sgcm_binding.json'):
                copy_new(RUN / name, RAW / name)
            copy_new(BASE / 'metadata/red_readiness_adapter/scene0029_spec.json', RAW / 'scene0029_spec.json')
            copy_new(BASE / 'tools/followup/contracts.py', RAW / 'contracts.py')
            for step in (10000, 20000, 30000):
                event_path = RUN / ('checkpoint_events/iteration_%06d.json' % step)
                event = read(event_path)
                copy_new(event_path, RAW / ('iteration_%06d.json' % step))
                copy_new(event['metrics'], RAW / ('eval_%d/metrics.json' % step))
                copy_new(Path(event['metrics']).parent / 'paper_region_metrics.json', RAW / ('eval_%d/paper_region_metrics.json' % step))
                copy_new(RUN / ('static_time_%d/summary.json' % step), RAW / ('static_time_%d_summary.json' % step))
                if step == 30000:
                    for frame in event['frames']:
                        copy_new(frame['hybrid_metrics'], RAW / 'eval_30000/frame_metrics' / Path(frame['hybrid_metrics']).name)
            contract_path = META / 'scene0029_10000_20000_30000_completion_contract.json'
            stage('three_node_original_contract_audit', [PY, str(TOOLS / 'audit_red_terminal_readonly.py')], contract_path)
            receipt = read(contract_path)
            assert receipt['status'] == 'FOLLOWUP_COMPLETE_CONTRACT_PASS' and not receipt['failure_fields']
            assert not receipt['owned_process_observation']['alive']
            stage('single_pass_training_extract', [PY, str(TOOLS / 'extract_red_training_rows_readonly.py')], RAW / 'sgcm_nps_training_rows.json')
            stage('saved_spec_terminal_summary', [PY, str(TOOLS / 'summarize_red_terminal_cpu.py'), str(META)])
            stage('real_saved_FP32_CPU_export', [PY, str(BASE / 'tools/gmc_saved_cpu_20261013_0905/export_saved_followup_cpu.py'),
                  '--source', str(RUN), '--output', str(REPORT), '--verified-contract', str(contract_path),
                  '--scene-label', '红车', '--architecture', str(TOOLS / 'architecture-components.svg')])
            for path in sorted(META.glob('scene0029_*.json')):
                copy_new(path, REPORT / path.name)
            # Keep exact originals plus one-pass training extraction without loading checkpoint payloads.
            shutil.copytree(RAW, REPORT / 'terminal_evidence')
            stage('real_saved_viewer_red_labels', [PY, str(TOOLS / 'build_red_saved_viewer.py'), '--output', str(REPORT)])
            stage('decode_and_embedded_report_verification', [PY, str(TOOLS / 'verify_red_saved_report.py'),
                  '--report', str(REPORT), '--js', str(REPORT / 'viewer_runtime.js'),
                  '--receipt', str(REPORT / 'asset_verification.json')])
            manifest = read(REPORT / 'manifest.json')
            terminal = read(META / 'scene0029_terminal_summary.json')
            accounting = read(META / 'scene0029_NPS6_SGCM145_accounting_summary.json')
            result = {'status': 'PASS_RED_30K_TERMINAL_AND_REAL_SAVED_CPU_REPORT', 'completed_utc': now(),
                      'collector_pid': os.getpid(), 'owned_training_pid': OWNER_PID, 'owned_training_exit_check': STATE['owned_training_exit_check'],
                      'report': str(REPORT), 'archive': str(REPORT.with_suffix('.tar.gz')), 'metadata': str(META),
                      'terminal': terminal['terminal'], 'summary': manifest['summary'], 'verification': manifest['verification'],
                      'NPS': terminal['NPS'], 'SGCM_totals': accounting['SGCM']['total_per_call_counters'],
                      'SGCM_calls': accounting['SGCM']['calls'], 'sampler': accounting['sampler'],
                      'restoration_verified': False, 'actual_model_Adam_RNG_restore_verified': False,
                      'human_verdict': None, 'GPU_used': False, 'training_started': False, 'continuous_training_polling': False,
                      'new_hash_or_fingerprint_computed': False, 'shared_gpu_lock_released_on_exit': True}
            write_new(REPORT / 'collector_result.json', result)
            with tarfile.open(str(REPORT.with_suffix('.tar.gz')), 'x:gz') as archive:
                archive.add(REPORT, arcname=REPORT.name)
            write_new(META / 'result.json', result)
            assert 'torch' not in sys.modules
            save(result['status'], result_json=str(META / 'result.json'), report_archive=str(REPORT.with_suffix('.tar.gz')))
    except BaseException as exc:
        if ACTIVE is not None and ACTIVE.poll() is None:
            ACTIVE.terminate()
            ACTIVE.wait(timeout=30)
        save('engineering_stopped_actual_partial_assets_preserved', error=repr(exc), active_cpu_child_pid=None)
        if not (META / 'failure.json').exists():
            write_new(META / 'failure.json', STATE)
        raise


if __name__ == '__main__':
    main()
