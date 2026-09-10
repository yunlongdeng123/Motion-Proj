"""按公开身份预留 V74 AV2 域内 FINAL，下载固定窗口的必要原始载荷。"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_worldsim_v73_av2_confirmation import listing,S5,SOURCE

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    config=ROOT/'configs/worldsim_v74/av2_final.json'
    evidence=ROOT/'docs/autoresearch/worldsim_v74/p0'
    data_root=Path('/root/autodl-tmp/data/worldsim_v74/av2_final_raw')
    started=time.monotonic()
    if config.exists():
        plan=json.loads(config.read_text())
    else:
        available=sorted(r['key'].rstrip('/').split('/')[-1] for r in listing(SOURCE) if r['type']=='directory')
        uuid=r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        refs=subprocess.run(['rg','-o','-i','--no-filename',uuid,'configs','docs','scripts'],cwd=ROOT,capture_output=True,text=True)
        if refs.returncode not in [0,1]:
            raise RuntimeError(refs.stderr)
        excluded=set(re.findall(uuid,refs.stdout.lower()))
        for split in ['train','val']:
            excluded.update(p.name for p in Path('/root/autodl-tmp/data/av2/sensor',split).iterdir() if p.is_dir())
        selected=[x for x in available if x not in excluded][:10]
        plan={'task':'WS-V74-P0-DATA-01','status':'identities_reserved','dataset':'AV2 Sensor','role':'FINAL',
              'official_source_split':'train','selected_log_ids':selected,'available_logs':len(available),
              'selection':'first 10 lexicographic public IDs absent from existing repository references and local raw logs; selected before payload listing',
              'build_lidar_indices':[5,15,20,30],'query_lidar_indices':[10,25],
              'training_policy':'AV2 FIT only from data_roles.json; this FINAL never enters shared training, hyperparameter selection or teacher generation',
              'payload_values_used_for_selection':False,'model_quality_read':False,'output':str(data_root),
              'source':'https://argoverse.github.io/user-guide/datasets/sensor.html'}
        write(config,plan)
    if 'logs' not in plan:
        logs=[]
        for identity in plan['selected_log_ids']:
            prefix=SOURCE+identity+'/'
            lidar=sorted((r for r in listing(prefix+'sensors/lidar/*') if r['type']=='file'),key=lambda r:r['key'])
            selected=[lidar[i] for i in sorted(plan['build_lidar_indices']+plan['query_lidar_indices'])]
            for name in ['annotations.feather','city_SE3_egovehicle.feather',
                         'calibration/egovehicle_SE3_sensor.feather','calibration/intrinsics.feather']:
                selected.extend(r for r in listing(prefix+name) if r['type']=='file')
            row={'log_id':identity,'available_lidar_sweeps':len(lidar),
                 'lidar_timestamps_ns':{str(i):int(Path(lidar[i]['key']).stem) for i in sorted(plan['build_lidar_indices']+plan['query_lidar_indices'])},
                 'files':[{'filename':r['key'][len(prefix):],'source':r['key'],'bytes':r['size']} for r in selected]}
            logs.append(row)
            print(json.dumps({'status':'planned','logs_done':len(logs),'log_id':identity}),flush=True)
        plan.update(status='raw_window_planned',logs=logs,planned_bytes=sum(r['bytes'] for log in logs for r in log['files']))
        write(config,plan)
    if args.download:
        commands=[]
        for log in plan['logs']:
            for item in log['files']:
                destination=data_root/log['log_id']/item['filename']
                destination.parent.mkdir(parents=True,exist_ok=True)
                if destination.exists() and destination.stat().st_size==item['bytes']:
                    continue
                commands.append('cp '+item['source']+' '+str(destination))
        command_file=ROOT/'work/v74_av2_final_copy.txt'
        command_file.write_text('\n'.join(commands)+'\n')
        with (ROOT/'work/v74_av2_final_copy.log').open('a') as output:
            if commands:
                subprocess.run([S5,'--no-sign-request','--numworkers','2','--retry-count','4','run',str(command_file)],
                    check=True,stdout=output,stderr=subprocess.STDOUT)
        missing=[str(data_root/log['log_id']/item['filename']) for log in plan['logs'] for item in log['files']
                 if not (data_root/log['log_id']/item['filename']).exists()]
        result={'status':'raw_ready' if not missing else 'missing_payload','logs':len(plan['logs']),
                'files':sum(len(log['files']) for log in plan['logs']),'planned_bytes':plan['planned_bytes'],
                'missing':missing,'wall_s':time.monotonic()-started,'model_quality_read':False,
                'annotation_or_lidar_values_opened':False,'coordinate_export':'pending; raw FINAL reserved separately from FIT/DEV',
                'data_root':str(data_root),'selected_log_ids':plan['selected_log_ids']}
        write(evidence/'av2_final_raw_summary.json',result)
        plan['status']=result['status']
        write(config,plan)
        print(json.dumps(result),flush=True)

if __name__=='__main__': main()
