"""Concrete eight-family GPU diagnostics, reusing each scene's frozen 30K.

Waits for missing source assets outside the GPU lock; never launches training.
Completed stages are reusable after an engineering failure, failed outputs stay.
"""
import argparse, json, os, subprocess, time, traceback
from pathlib import Path
from native_scene_controller import same_process
from resource_control import require_start, stop_requested

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
PY = '/root/autodl-tmp/envs/drivingeditor/bin/python'
VIDEO_PY = '/root/autodl-tmp/envs/motionproj/bin/python'
FAMILIES = ['ego_speed', 'ego_lane_change', 'ego_direction', 'vehicle_speed',
            'vehicle_lane_change', 'vehicle_direction', 'vehicle_removal']


def read(path):
    return json.loads(Path(path).read_text())


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().split()[21]


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    parser.add_argument('--resume-failed', action='store_true')
    parser.add_argument('--ego-speed-clock', choices=['fixed_world', 'global_retime_control'])
    args = parser.parse_args()
    require_start()
    dest = ROOT/'benchmark_execution_v1'/args.context
    previous = None
    if args.resume_failed:
        previous = read(dest/'status.json')
        assert previous['phase'] == 'engineering_repair_needed_preserved'
        assert not same_process(previous['pid'], previous['start_ticks'])
        child = previous.get('child')
        assert not child or not same_process(child['pid'], child['start_ticks'])
    else:
        dest.mkdir(parents=True, exist_ok=False)
    clock_policy = previous.get('ego_speed_clock_policy', 'global_retime_control') if previous else (args.ego_speed_clock or 'fixed_world')
    if previous and args.ego_speed_clock:
        assert args.ego_speed_clock == clock_policy, 'Resume must preserve completed-stage clock policy; use a separate control for policy changes'
    attempt = previous['attempt']+1 if previous else 1
    directory = dest/f'attempt_{attempt}'
    directory.mkdir(exist_ok=False)
    if previous:
        write(directory/'previous_status.json', previous)
    state = dict(context=args.context, pid=os.getpid(), start_ticks=ticks(os.getpid()),
                 phase='waiting_destination30k', child=None, attempt=attempt,
                 stages=previous['stages'] if previous else {},
                 completed_native_families=[], completed_fixed_backend_families=[],
                 formal_admitted=0, quality_accepted=False, human_verdict=None,
                 new_training_updates=0, ego_speed_clock_policy=clock_policy)

    def update(**kw):
        state.update(kw)
        write(dest/'status.json', state)

    def wait30k(context):
        while True:
            require_start()
            receipt = ROOT/'native_scene_training_v1'/context/'formal30k/returned.json'
            if receipt.exists():
                returned = read(receipt)
                assert returned['iteration'] == 30000 and returned['reason'] == 'native_returned'
                return
            time.sleep(5)

    def stage(name, command, summary):
        old = state['stages'].get(name)
        if old:
            assert Path(old['summary']).exists()
            return Path(old['summary'])
        require_start()
        update(phase=name, command=command)
        with (directory/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(child=dict(pid=child.pid, start_ticks=ticks(child.pid)))
            while child.poll() is None:
                if stop_requested():
                    child.terminate()
                    child.wait(timeout=45)
                    raise RuntimeError('Quota stop: partial read-only results preserved')
                time.sleep(3)
        update(child=None, returncode=child.returncode)
        assert child.returncode == 0 and summary.exists(), 'Stage failed; preserve attempt for agent engineering repair'
        state['stages'][name] = dict(summary=str(summary))
        update()
        return summary

    update()
    try:
        wait30k(args.context)
        base = [PY, str(ROOT/'code/render_benchmark_suite.py'), '--context', args.context, '--ego-speed-clock', clock_policy]
        entry = directory/'camera_object_entry'
        proof = stage('camera_object_entry', base+['--output', str(entry), '--families']+FAMILIES+['--single-frame'], entry/'summary.json')
        assert read(proof)['adapter_actual_native_max_difference'] <= 2e-6
        full = directory/'camera_objects'
        proof = stage('camera_objects', base+['--output', str(full), '--families']+FAMILIES, full/'summary.json')
        completed = read(proof)['completed']
        update(completed_native_families=[x['family'] for x in completed])
        for item in completed:
            family = item['family']
            if not family.startswith('vehicle_'):
                continue
            source = Path(item['source_request'])
            batch = proof.parent
            summary = batch/family/'propainter/summary.json'
            stage('backend_'+family, [VIDEO_PY, str(ROOT/'code/inpaint_baseline.py'), '--root', str(batch), '--case', family, '--input-kind', 'native_3d'], summary)
            state['completed_fixed_backend_families'].append(family)
            update()
        insertion = read(ROOT/'benchmark_cases_v1/scenes'/args.context/'requests/vehicle_insertion.json')
        if insertion['input_screen_eligible']:
            source_context = insertion['source_asset_context']
            update(phase='waiting_distinct_source_asset30k', source_asset_context=source_context)
            wait30k(source_context)
            entry = directory/'foreign_asset_entry'
            proof = stage('foreign_asset_entry', base+['--output', str(entry), '--families', 'vehicle_insertion', '--single-frame'], entry/'summary.json')
            assert read(proof)['adapter_actual_native_max_difference'] <= 2e-6
            full = directory/'foreign_asset'
            proof = stage('foreign_asset', base+['--output', str(full), '--families', 'vehicle_insertion'], full/'summary.json')
            state['completed_native_families'] += [x['family'] for x in read(proof)['completed']]
        update(phase='CPU_delivery')
        stage('CPU_delivery', [PY, str(ROOT/'code/analyze_benchmark_suite.py'), '--execution', str(dest)], dest/'cpu_delivery/summary.json')
        update(phase='diagnostic_execution_complete_formal_admission_pending',
               camera_generation_status='unknown_domain_masks_pending_native_only',
               insertion_generation_status='no_old_hole; imported asset protected; no fake backend call')
    except Exception:
        update(phase='resource_stop' if stop_requested() else 'engineering_repair_needed_preserved', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
