"""CPU materialization of remaining dev and reserved contexts, not paper results.

Extract original RGB/LiDAR/calibration once per scene; selected windows need a
fresh quality check. Existing healthy native scenes and split assignments stay
unchanged. No editing outputs are read to select data.
"""
import concurrent.futures as futures
import datetime,json,os,subprocess,time,traceback
from pathlib import Path
from prepare_motion_sensor_inputs import ROOT,worker,write
from resource_control import require_start,stop_requested

PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
DEST=ROOT/'benchmark_expansion_v1'

def ready_scene(item):
    require_start()
    scene,index=item
    result=worker(scene,index)
    require_start()
    command=[PY,str(ROOT/'code/prepare_benchmark_cases.py'),'--context',scene['context']]
    subprocess.run(command,check=True)
    return dict(context=scene['context'],split=scene['split'],frames=len(result['frames']),real_lidar_points=sum(x['lidar_points'] for x in result['frames']),folder=str(ROOT/'motion_scene_inputs_v1'/scene['context']))

def main():
    require_start();DEST.mkdir(exist_ok=True)
    assert not (DEST/'dispatch.json').exists(),'Unique CPU preparation owner; inspect existing PID before recovery'
    dispatch=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_workers=3,uses_gpu=False)
    write(DEST/'dispatch.json',dispatch)
    manifest=json.loads((ROOT/'evaluation_set_v1/manifest.json').read_text())
    proposals={x['context']:x for x in json.loads((ROOT/'evaluation_admission_v2/operation_window_proposals.json').read_text())['proposals']}
    selected=[];existing=[]
    for index,entry in enumerate(sorted(manifest['scenes'],key=lambda x:x['context']),9100):
        if entry['split'].startswith('previous_'):continue
        folder=ROOT/'motion_scene_inputs_v1'/entry['context'];proposal=proposals[entry['context']]
        if (folder/'sensor_preparation.json').exists():existing.append(entry['context']);continue
        target=proposal.get('moving_target_speed_window') or proposal.get('motion_window')
        scene=dict(entry)
        if target:scene.update(selected_window=target['window'],selected_track=target['target_track'],target_stats=target,selection_policy='motion/speed proposal chosen before editing; new window requires independent rescreen')
        else:scene.update(target_stats=None,selection_policy='delete/ego-only fallback: no motion-qualified target; never count as speed/turn-ready')
        selected.append((scene,index))
    # Dev inputs first; frozen reserved inputs are prepared without inspecting
    # edited outputs or changing splits. Optical degradation remains a stratum.
    selected.sort(key=lambda x:(x[0]['split']=='heldout_reserved',x[0]['context']))
    write(DEST/'input_queue.json',dict(existing=existing,scenes=[s for s,_ in selected],does_not_use_edit_outputs=True,split_changes=0))
    status=dict(**dispatch,phase='real_sensor_and_case_preparation',target_contexts=len(selected),existing_contexts=existing,completed=[],errors=[],quality_accepted=False,human_verdict=None)
    write(DEST/'status.json',status)
    for context in existing:
        subprocess.run([PY,str(ROOT/'code/prepare_benchmark_cases.py'),'--context',context],check=True)
    with futures.ProcessPoolExecutor(max_workers=3) as pool:
        jobs={pool.submit(ready_scene,item):item[0]['context'] for item in selected}
        for future in futures.as_completed(jobs):
            context=jobs[future]
            try:status['completed'].append(future.result())
            except Exception:status['errors'].append(dict(context=context,error=traceback.format_exc()))
            status['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(DEST/'status.json',status)
            if stop_requested():
                for job in jobs:job.cancel()
                status['phase']='quota_stop_partial_inputs_preserved';write(DEST/'status.json',status);return
    status['phase']='inputs_and_cases_materialized' if not status['errors'] else 'engineering_errors_preserved';write(DEST/'status.json',status)
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
