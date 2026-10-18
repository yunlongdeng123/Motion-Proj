"""一次性30k评价与六项编辑；默认只读CPU预检，不排队抢占或自动重试。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
POST = BASE / 'post30k_r3'
RUN = BASE / 'runs/gmc_two_actor_fixed30k_r1'
SPEC = BASE / 'specs/scene0008_023_071.json'
LOCK = BASE.parent / 'gpu_slot.lock'
REFERENCE = BASE.parent / 'followup_scene_prep_20261013/reports/gmc_gn11_30k_saved_cpu_20261013_0910/manifest.json'
TRACKS = ('NoQttdyd4C16Y_zkGhiFKg', 'fXVhuN-RuJDvZwnkQ-0YSQ')
FIELDS = {'background', 'instances', 'environment', 'sampler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write_status(value):
    value.update(pid=os.getpid(), utc=datetime.now(timezone.utc).isoformat())
    path = POST / 'status.json.partial'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(path, POST / 'status.json')

def stages():
    checkpoint = str(RUN / 'checkpoint_030000.pt')
    common = ['--spec', str(SPEC), '--checkpoint', checkpoint, '--expected-step', '30000', '--gpu-lock', str(LOCK)]
    yield 'scene_evaluation', [str(BASE / 'infra/evaluate_scene.py'), *common, '--fixed-camera-frame', '0008033',
                               '--output', str(POST / 'evaluation'), '--run'], POST / 'evaluation', True
    for actor, track in enumerate(TRACKS):
        for operation in ('delete', 'translate', 'rotate'):
            name = 'actor%d_%s' % (actor, operation)
            output = POST / 'edits' / name
            command = [str(BASE / 'infra/render_edits.py'), *common, '--track', track, '--operation', operation,
                       '--output', str(output), '--run']
            if operation == 'translate':
                command += ['--translate-local-m', '0.5', '0', '0']
            if operation == 'rotate':
                command += ['--rotate-degrees', '30']
            yield name, command, output, True
    yield 'CPU_edit_contract', [str(BASE / 'infra/collect_edit_summary.py'), '--edit-root', str(POST / 'edits'),
         '--prepared', str(BASE / 'prepared/scene0008_two_actor_v1'), '--expected-steps', '30000',
         '--reference-manifest', str(REFERENCE), '--output', str(POST / 'edit_summary.json')], POST, False

def validate_inputs():
    status = read(RUN / 'status.json')
    if status.get('state') != 'complete' or status.get('step') != 30000:
        raise ValueError('必须是真实30k训练终态，不重复训练')
    spec = read(SPEC)
    if spec.get('scene_id') != 'scene0008_023_071' or not REFERENCE.is_file():
        raise ValueError('实际GMC scene或GT reference缺失')
    proofs = []
    for step in (10000, 20000, 30000):
        path = BASE / 'post30k_r1' / ('restore_%06d' % step) / 'receipt.json'
        proof = read(path)
        if (proof.get('step') != step or proof.get('checkpoint') != str(RUN / ('checkpoint_%06d.pt' % step))
                or proof.get('runtime_restoration_verified') is not True
                or proof.get('scope') != 'runtime_model_Adam_and_RNG_tensors_restored_before_first_update'
                or set(proof.get('checked_fields', [])) != FIELDS or proof.get('optimizer_updates') != 0
                or not str(proof.get('execution_device', '')).startswith('cuda')):
            raise ValueError('既有真实GPU恢复凭据绑定不完整: ' + str(path))
        proofs.append(str(path))
    for source in ('evaluate_scene.py', 'render_edits.py', 'collect_edit_summary.py', 'priority.py'):
        if not (BASE / 'infra' / source).is_file():
            raise FileNotFoundError(source)
    if POST.exists():
        raise FileExistsError('保留原attempt，拒绝重复r3或覆盖: ' + str(POST))
    return {'status': 'CPU_POST30K_INPUTS_VALIDATED_GPU_NOT_STARTED', 'Torch_imported': 'torch' in sys.modules,
            'prior_verified_restore_steps': [10000, 20000, 30000], 'prior_restore_receipts': proofs,
            'GPU_stages': 7, 'CPU_summary_stages': 1, 'optimizer_updates': 0, 'output_created': False}

def verify_owned_processes_exited():
    # PID若被复用只记录，不终止；真实旧入口仍在则停止新launch。
    signatures = {37747: ('/infra/train.py', str(RUN)), 65212: ('/infra/queue_post30k.py',),
                  65834: ('/infra/queue_post30k_r2.py',), 65835: ('/infra/evaluate_scene.py', str(BASE / 'post30k_r2'))}
    observed = []
    for pid, fragments in signatures.items():
        path = Path('/proc') / str(pid) / 'cmdline'
        try:
            command = path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            observed.append({'pid': pid, 'old_owned_process_exited': True})
            continue
        if all(fragment in command for fragment in fragments):
            raise RuntimeError('旧owned进程仍在，不重复启动: ' + str(pid))
        observed.append({'pid': pid, 'PID_reused_by_unrelated_process': True, 'other_process_modified': False})
    return observed

def verify_stage(name):
    if name == 'scene_evaluation':
        result = read(POST / 'evaluation/scene_evaluation_manifest.json')
        contract = result.get('contracts', {})
        if (result.get('status') != 'SCENE_EVALUATION_EXPORTED' or result.get('spec') != read(SPEC)
                or result.get('checkpoint') != {'path': str(RUN / 'checkpoint_030000.pt'), 'step': 30000}
                or result.get('instance_ids') != list(TRACKS)
                or [contract.get(key) for key in ('observed_frames', 'fixed_camera_frames', 'FP32_frame_files')] != [49, 49, 98]
                or contract.get('model_and_optimizer_unmodified_after_evaluation') is not True
                or contract.get('optimizer_updates') != 0 or contract.get('all_FP32_finite') is not True
                or any(len(result['streams'][stream]['frames']) != 49 for stream in ('observed', 'fixed_camera'))):
            raise ValueError('完整49+49真实评价合同缺失')
    elif name == 'CPU_edit_contract':
        result = read(POST / 'edit_summary.json')
        contract = result.get('engineering_contract', {})
        if (contract.get('passed') is not True or contract.get('verified_FP32_pairs') != 294
                or contract.get('GT_reference_PNG_verified') is not True or contract.get('GT_reference_max_byte_delta') != 0
                or result.get('source', {}).get('checkpoint_steps') != 30000
                or result.get('instance_ids') != list(TRACKS)):
            raise ValueError('独立CPU实际294对编辑/GT合同缺失')
    else:
        actor, operation = int(name[5]), name.split('_', 1)[1]
        result = read(POST / 'edits' / name / 'edit_receipt.json')
        arguments = result.get('arguments', {})
        if (result.get('status') != 'EDIT_RENDER_EXPORTED_VISUAL_REVIEW_PENDING'
                or result.get('actual_checkpoint_step') != 30000 or result.get('spec') != read(SPEC)
                or arguments.get('expected_step') != 30000 or arguments.get('checkpoint') != str(RUN / 'checkpoint_030000.pt')
                or arguments.get('track') != TRACKS[actor] or arguments.get('operation') != operation
                or len(result.get('frames', [])) != 49):
            raise ValueError('编辑真实30k/scene/ID/operation/49frames绑定不完整: ' + name)

def worker():
    sys.path.insert(0, str(BASE / 'infra'))
    from priority import inspect_priority
    completed = []
    try:
        for name, arguments, output, GPU in stages():
            if GPU:
                priority = inspect_priority()
                if not priority['allowed']:
                    write_status({'state': 'yielded_to_priority_before_stage', 'current': name, 'priority': priority,
                                  'completed': completed, 'automatic_retries': False, 'assets_preserved': True})
                    return 3
            command = ['nice', '-n', '10', sys.executable, '-B', '-u'] + arguments
            threads = '4' if GPU else '1'
            with (POST / (name + '.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env=dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads))
                write_status({'state': 'running_serial_postprocessing', 'current': name, 'child_pid': child.pid,
                              'command': command, 'completed': completed, 'optimizer_updates': 0, 'other_tasks_modified': False})
                code = child.wait()
            if code:
                reason = read(output / 'failure.json') if (output / 'failure.json').is_file() else {}
                yielded = (output / 'priority_yield.json').is_file() or 'StreetCrafter' in reason.get('error', '')
                write_status({'state': 'priority_yielded_assets_preserved' if yielded else 'engineering_failed_assets_preserved',
                              'current': name, 'returncode': code, 'completed': completed, 'reason': reason,
                              'automatic_retries': False, 'optimizer_updates': 0})
                return code
            verify_stage(name)
            completed.append(name)
        write_status({'state': 'POST30K_EVALUATION_SIX_EDITS_CPU_CONTRACT_PASS_VISUAL_REVIEW_PENDING',
                      'completed': completed, 'prior_verified_restore_steps': [10000, 20000, 30000],
                      'GPU_restore_executed_in_this_attempt': False, 'human_quality_verdict': None,
                      'optimizer_updates': 0, 'other_tasks_modified': False})
        return 0
    except BaseException as error:
        write_status({'state': 'engineering_failed_assets_preserved', 'error': repr(error), 'completed': completed,
                      'automatic_retries': False, 'optimizer_updates': 0})
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker())
    preflight = validate_inputs()
    if not args.launch:
        print(json.dumps(preflight, ensure_ascii=False))
        return
    observed = verify_owned_processes_exited()
    sys.path.insert(0, str(BASE / 'infra'))
    from priority import require_gpu_slot
    slot, admission = require_gpu_slot(LOCK)
    try:
        # 忙时在创建POST之前退出；实际child仍独立取得锁并再检查资源。
        POST.mkdir()
    finally:
        slot.close()
    command = ['nice', '-n', '10', sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker']
    with (POST / 'controller.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), start_new_session=True)
    receipt = dict(preflight, status='POST30K_CONTROLLER_LAUNCHED', pid=child.pid, command=command, utc=datetime.now(timezone.utc).isoformat(),
                   admission=admission, owned_exit_snapshot=observed, parent_GPU_use=False,
                   automatic_retries=False, output_created=True, other_tasks_modified=False)
    (POST / 'launch.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()
