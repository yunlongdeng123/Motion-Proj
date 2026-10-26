"""Ego-only speed versus global playback control; same request, no new cases."""
import argparse, json, os, subprocess, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from benchmark_geometry import ROOT, read, edit_clocks
from resource_control import require_start, stop_requested
from native_scene_controller import same_process


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', required=True)
    p.add_argument('--resume-failed', action='store_true')
    args = p.parse_args()
    dest = ROOT/'ego_speed_clock_control_v1'/args.context
    previous = None
    if args.resume_failed:
        previous = read(dest/'status.json')
        assert previous['phase'] == 'engineering_repair_needed_preserved'
        assert not same_process(previous['pid'], previous['start_ticks'])
        child = previous.get('child')
        assert not child or not same_process(child['pid'], child['start_ticks'])
    else:
        dest.mkdir(parents=True, exist_ok=False)
    attempt = previous.get('attempt', 1)+1 if previous else 1
    directory = dest/f'attempt_{attempt}'
    directory.mkdir(exist_ok=False)
    if previous:
        (directory/'previous_status.json').write_text(json.dumps(previous, indent=2))
    old_output = Path(previous.get('native_output', str(dest/'fixed_world'))) if previous else None
    native_output = old_output if old_output is not None and (old_output/'summary.json').exists() else directory/'fixed_world'
    state = dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
                 context=args.context, phase='waiting_global_playback_reference', child=None,
                 new_training_updates=0, additional_case_count=0, attempt=attempt, native_output=str(native_output))

    def update(**kw):
        state.update(kw)
        tmp = dest/'status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(dest/'status.json')

    update()
    try:
        while True:
            require_start()
            parent = read(ROOT/'benchmark_execution_v1'/args.context/'status.json')
            stage = parent['stages'].get('camera_objects')
            if stage:
                reference = Path(stage['summary']).parent/'ego_speed'
                assert read(reference/'summary.json')['ego_speed_clock_policy'] == 'global_retime_control'
                break
            if parent['phase'] in ['engineering_repair_needed_preserved', 'resource_stop']:
                raise RuntimeError('Reference failed; preserve control for engineering continuation')
            time.sleep(5)
        require_start()
        command = ['/root/autodl-tmp/envs/drivingeditor/bin/python', str(ROOT/'code/render_benchmark_suite.py'),
                   '--context', args.context, '--output', str(native_output), '--families', 'ego_speed', '--ego-speed-clock', 'fixed_world']
        update(phase='GPU_fixed_world_ego_speed', command=command, reference=str(reference))
        with (directory/'native.log').open('x') as log:
            if (native_output/'summary.json').exists():
                child = None
            else:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if child is not None:
                update(child=dict(pid=child.pid, start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]))
                while child.poll() is None:
                    if stop_requested():
                        child.terminate()
                        child.wait(timeout=45)
                        raise RuntimeError('Resource stop; partial control preserved')
                    time.sleep(3)
                update(child=None, returncode=child.returncode)
                assert child.returncode == 0
        native = native_output/'ego_speed'
        fixed = read(native/'summary.json')
        original = read(reference/'summary.json')
        batch = read(native_output/'summary.json')
        assert batch['adapter_actual_native_max_difference'] <= 2e-6 and batch['master_files_and_tensor_versions_unchanged']
        request = read(ROOT/'benchmark_cases_v1/scenes'/args.context/'requests/ego_speed.json')
        rows = []
        for i, row in enumerate(request['frames']):
            expected = edit_clocks(row, 'ego_speed', 'fixed_world')
            actual = fixed['rows'][i]
            old = original['rows'][i]
            assert actual['world_model_time'] == expected['world_model_time'] == row['model_time']
            assert actual['dynamic_query_model_time'] == row['model_time']
            assert actual['camera_path_query_model_time'] == old['camera_path_query_model_time'] == row['query_model_time']
            name = f'{row["raw"]:03d}'
            a = np.load(native/'float'/(name+'.npz'))['rgb']
            b = np.load(reference/'float'/(name+'.npz'))['rgb']
            rows.append(dict(raw=row['raw'], camera_query_model_time=row['query_model_time'],
                             fixed_world_time=row['model_time'], global_control_world_time=old['world_model_time'],
                             full_frame_fixed_vs_global_MAE=float(np.abs(a-b).mean())))
        frames = sorted((native/'native_rgb').glob('*.png'))
        middle = frames[3*len(frames)//4]
        sheet = Image.new('RGB', (960, 350), '#152b3d')
        for j, (folder, label) in enumerate([(reference, 'Global playback control'), (native, 'Ego-only speed / fixed world clock')]):
            sheet.paste(Image.open(folder/'native_rgb'/middle.name), (j*480, 30))
            ImageDraw.Draw(sheet).text((j*480+8, 8), label+' / raw'+middle.stem, fill='white')
        sheet.save(dest/'comparison.jpg', quality=95)
        result = dict(context=args.context, status='clock_separation_GPU_complete', rows=rows,
                      same_camera_requests=True, fixed_world_other_actor_times_preserved=True,
                      global_playback_is_legacy_control_not_ego_only_edit=True,
                      mean_RGB_difference_not_quality=float(np.mean([x['full_frame_fixed_vs_global_MAE'] for x in rows])),
                      actual_native_adapter_max_difference=batch['adapter_actual_native_max_difference'],
                      GPU_seconds=batch['seconds'], cuda_peak_allocated=batch['cuda_peak_allocated'], cuda_peak_reserved=batch['cuda_peak_reserved'],
                      new_training_updates=0, additional_case_count=0, counterfactual_RGB_GT=False,
                      formal_admission=0, quality_accepted=False, human_verdict=None)
        (dest/'summary.json').write_text(json.dumps(result, indent=2))
        update(phase='complete')
    except Exception:
        update(phase='resource_stop' if stop_requested() else 'engineering_repair_needed_preserved', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
