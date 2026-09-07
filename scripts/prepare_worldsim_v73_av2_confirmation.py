"""Select new external logs using identities/timestamps only; optionally copy exact sensor files.

No image, point cloud, annotation values or model scores are read here. Existing
AV2 development logs are used separately to implement the format bridge.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
S5='/root/autodl-tmp/bin/s5cmd'
SOURCE='s3://argoverse/datasets/av2/sensor/train/'
RING=['ring_front_center','ring_front_left','ring_front_right','ring_side_left',
      'ring_side_right','ring_rear_left','ring_rear_right']
BUILD=[5,15,20,30]
EVAL=[10,25]
UUID=r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'


def listing(url):
    proc=subprocess.run([S5,'--json','--no-sign-request','--retry-count','4','ls',url],
                        check=True,capture_output=True,text=True)
    return [{k:row[k] for k in ['key','size','type'] if k in row}
            for line in proc.stdout.splitlines() if line.strip()
            for row in [json.loads(line)]]


def plan_log(log_id):
    prefix=SOURCE+log_id+'/'
    rows=listing(prefix+'*')
    files={row['key'][len(prefix):]:row for row in rows if row['type']=='file'}
    lidar=sorted(name for name in files if name.startswith('sensors/lidar/') and name.endswith('.feather'))
    chosen={}; cameras=[]; missing=[]; timestamps={}
    for index in BUILD+EVAL:
        if index>=len(lidar):
            missing.append({'kind':'lidar_index','index':index}); continue
        name=lidar[index]; chosen[name]=files[name]
        timestamps[index]=int(Path(name).stem)
    for channel in RING:
        available=sorted(name for name in files if name.startswith('sensors/cameras/'+channel+'/') and name.endswith('.jpg'))
        for index in BUILD:
            if index not in timestamps or not available:
                missing.append({'kind':'camera','channel':channel,'lidar_index':index}); continue
            name=min(available,key=lambda name:abs(int(Path(name).stem)-timestamps[index]))
            camera_ts=int(Path(name).stem)
            chosen[name]=files[name]
            cameras.append({'channel':channel,'lidar_index':index,'timestamp_ns':camera_ts,
                            'delta_from_lidar_ns':camera_ts-timestamps[index],'filename':name})
    for name in ['annotations.feather','city_SE3_egovehicle.feather',
                 'calibration/egovehicle_SE3_sensor.feather','calibration/intrinsics.feather']:
        if name in files: chosen[name]=files[name]
        else: missing.append({'kind':'metadata_file','filename':name})
    return {'log_id':log_id,'role':'external_confirmation','source_split':'train',
            'available_lidar_sweeps':len(lidar),'build_lidar_indices':BUILD,'evaluation_lidar_indices':EVAL,
            'lidar_timestamps_ns':timestamps,'build_camera_observations':cameras,'missing_metadata':missing,
            'files':[{'filename':name,'source':row['key'],'bytes':row['size']} for name,row in sorted(chosen.items())],
            'planned_bytes':sum(row['size'] for row in chosen.values())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'configs/worldsim_v73/av2_external_confirmation_r1.json')
    parser.add_argument('--dataset-root',type=Path,default=Path('/root/autodl-tmp/data/av2/sensor/train'))
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args(); start=time.monotonic()
    if args.output.exists():
        plan=json.loads(args.output.read_text())
    else:
        entries=listing(SOURCE)
        available=sorted(row['key'].rstrip('/').split('/')[-1] for row in entries if row['type']=='directory')
        proc=subprocess.run(['rg','-o','-i','--no-filename',UUID,'configs','docs','scripts'],
                            cwd=ROOT,capture_output=True,text=True)
        if proc.returncode not in [0,1]: raise RuntimeError(proc.stderr)
        recorded=set(re.findall(UUID,proc.stdout.lower()))
        local={p.name for split in ['/root/autodl-tmp/data/av2/sensor/val',str(args.dataset_root)]
               if Path(split).exists() for p in Path(split).iterdir() if p.is_dir()}
        excluded=recorded|local
        candidates=[log for log in available if log not in excluded]
        selected=candidates[:20]
        # Identity selection is complete before any selected-log object listing.
        plan={'status':'identities_selected','dataset':'AV2 Sensor','source_split':'train',
              'role':'external_confirmation','selection':'first 20 lexicographic official train log IDs absent from repository identity references and local log directories',
              'available_logs':len(available),'excluded_available_logs':len(set(available)&excluded),
              'candidate_logs':len(candidates),'selected_log_ids':selected,
              'build_lidar_indices':BUILD,'evaluation_lidar_indices':EVAL,'camera_channels':RING,
              'payload_values_read_for_selection':False,'model_scores_used':False,
              'shared_training_allowed':False,'hyperparameter_selection_allowed':False,
              'scope':'external new-log AND sensor-domain confirmation; not same-distribution nuScenes confirmation',
              'missing_policy':'retain selected logs and missing input/return denominators; do not replace by quality',
              'method_status':'not selected yet; use already exposed AV2 logs for format development',
              'sources':['https://argoverse.github.io/user-guide/getting_started.html',
                         'https://github.com/argoverse/user-guide/blob/main/guide/src/datasets/sensor.md']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(plan,indent=2)+'\n')
    if 'logs' not in plan:
        with ThreadPoolExecutor(max_workers=4) as pool:
            plan['logs']=list(pool.map(plan_log,plan['selected_log_ids']))
        plan.update(status='payload_manifest_ready',planning_wall_s=time.monotonic()-start,
                    planned_bytes=sum(row['planned_bytes'] for row in plan['logs']))
        args.output.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:plan[k] for k in ['status','available_logs','candidate_logs','selected_log_ids','planned_bytes'] if k in plan}),flush=True)
    if args.download:
        state=Path('/root/autodl-tmp/controller_logs/v73_av2_confirmation')
        state.mkdir(parents=True,exist_ok=True)
        commands=[]
        for row in plan['logs']:
            for item in row['files']:
                destination=args.dataset_root/row['log_id']/item['filename']
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.is_file() and destination.stat().st_size==item['bytes']: continue
                commands.append('cp '+item['source']+' '+str(destination))
        command_file=state/'copy_commands.txt'
        command_file.write_text('\n'.join(commands)+'\n')
        status={'status':'copying','files_requested':len(commands),'started_unix_s':time.time()}
        (state/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        with (state/'copy.log').open('a') as log:
            proc=subprocess.run([S5,'--no-sign-request','--numworkers','4','--retry-count','6',
                                 'run',str(command_file)],stdout=log,stderr=subprocess.STDOUT) if commands else None
        status.update(status='done' if proc is None or proc.returncode==0 else 'failed',
                      returncode=proc.returncode if proc else 0,finished_unix_s=time.time(),
                      sensor_values_opened=False)
        (state/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print(json.dumps(status),flush=True)
        if status['returncode']: raise SystemExit(status['returncode'])


if __name__=='__main__': main()
