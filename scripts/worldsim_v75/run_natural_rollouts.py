"""单个真实残余案例的五个固定输入比较；前置失败或OOM即停。"""
import copy
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime,timezone
import numpy as np
from prepare_argoverse import project_track

P=Path('/root/autodl-tmp/motion_proj');S=P/'scripts/worldsim_v75'
SOURCE=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-SOURCES-01/20260920-r1')
LOG='0bae3b5e-417d-3b03-abaa-806b433233b8'
BASE=SOURCE/'cases'/LOG/'base'
READ=SOURCE/'cases'/LOG/'reconstruction'
OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-ROLLOUT-01/20260920-r1')
VARIANTS=['gt_clean','dvgt_metric','dvgt_lidar_scaled','ordinary_bbox','reference_lidar']

def call(script,log,args):
    command=['bash','-lc','source scripts/worldsim_v75/environment.sh && exec python '+script+' '+ ' '.join(map(str,args))]
    # 所有参数只来自本脚本生成的固定无空格路径、枚举和数字。
    with log.open('w') as f:
        result=subprocess.run(command,cwd=P,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{script}: rc={result.returncode}, stopped; log={log}')

def verify_condition(case,target):
    clean=np.load(OUT/'gt_clean/conditions.npy',mmap_mode='r');other=np.load(case/'conditions.npy',mmap_mode='r')
    trajectory=np.load(BASE/'trajectory.npz')
    reference=next(t for t in json.loads((BASE/'scene.json').read_text())['tracks'] if t['id']==target['id'])
    rows=[]
    for f in range(61):
        changed=np.any(clean[f]!=other[f],axis=-1);allowed=np.zeros((704,1280),bool)
        for t in [reference,target]:
            projection=project_track(t,f,trajectory['camera_world'],trajectory['K'])
            assert projection is not None
            b=np.array(projection['bounds']);lo=np.maximum(np.floor(b[:2]-8).astype(int),0);hi=np.minimum(np.ceil(b[2:]+8).astype(int),[1280,704])
            if (hi>lo).all():allowed[lo[1]:hi[1],lo[0]:hi[0]]=True
        outside=int((changed&~allowed).sum());assert outside==0,(case.name,f,outside)
        rows.append({'frame':f,'changed_pixels':int(changed.sum()),'outside_target_union':outside})
    assert any(r['changed_pixels'] for r in rows)
    (case/'condition_binding_check.json').write_text(json.dumps({'status':'passed','checked_frames':61,'rows':rows,
        'other_scene_components_exactly_unchanged':True,'claim':'geometry/raster binding only; no measured generated consequence yet'},indent=2)+'\n')

def main():
    assert not OUT.exists(),'拒绝重跑/覆盖此发现比较'
    assert json.loads((SOURCE/'readout_queue_result.json').read_text())['status']=='complete'
    read=json.loads((READ/'readout_ray_control_result.json').read_text());assert read['generation_admitted']
    assert read['readouts']['dvgt_metric']['center_error_m']>2*read['readouts']['reference_lidar']['center_error_m']
    OUT.mkdir(parents=True)
    protocol={'task_id':'WS-V75-NATURAL-ROLLOUT-01','run_id':'20260920-r1','frozen_utc':datetime.now(timezone.utc).isoformat(),
              'log_id':LOG,'target':read['target'],'base_dir':str(BASE),'readout_dir':str(READ),'variants':VARIANTS,'seed':42,
              'selection':'discovery follow-up to observed raw residual 2.96m; other three sources retained; not an independent test',
              'input':'initial real RGB + same text, same GT ego/map/other actors/future displacements; target initial translation from selected readout',
              'extra_information':['GT dimensions/yaw and future actor/ego trajectories shared by all arms','LiDAR scale or target readout adds real metric observations in named diagnostic arms'],
              'intervention':'constant initial world translation error across full target track; from frame0; same initial RGB and RNG',
              'frames':237,'fps':30,'primary_evaluation_frames':[0,15,30,45,60],
              'evaluator':'existing FasterRCNN, score .25/IoU .3; real-frame calibration ±4px <=2px and all five matches before generation',
              'stop':'preflight/render/evaluation failure or any OOM stops controller; no automatic retries/downsizing',
              'claim_boundary':'conditional state-readout diagnostic, not pure visual end-to-end simulator; no policy feedback; no universal failure claim',
              'human_verdict':None}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    result={'status':'running','pid':os.getpid(),'completed':[],'human_verdict':None};began=time.monotonic()
    try:
        print(json.dumps({'stage':'real_evaluator_preflight'}),flush=True)
        call('scripts/worldsim_v75/evaluate_natural_rollout.py',OUT/'evaluator_prepare.log',['prepare','--run-dir',OUT])
        original=json.loads((BASE/'scene.json').read_text())
        manifest=json.loads((BASE/'input_manifest.json').read_text())
        for name in VARIANTS:
            print(json.dumps({'stage':'render','variant':name}),flush=True)
            case=OUT/name;case.mkdir()
            data=copy.deepcopy(original)
            target=next(t for t in data['tracks'] if t['id']==read['target'])
            offset=np.zeros(3) if name=='gt_clean' else np.array(read['readouts'][name]['offset_world_m'])
            target['centers']=(np.array(target['centers'])+offset).tolist()
            assert all(a==b for a,b in zip(data['tracks'],original['tracks']) if a['id']!=target['id'])
            assert data['lines']==original['lines'] and data['crossings']==original['crossings']
            (case/'scene.json').write_text(json.dumps(data,indent=2)+'\n')
            m={**manifest,'task_id':protocol['task_id'],'run_id':protocol['run_id'],'variant':name,
               'state_source':name,'role':'discovery_conditional_state_readout_diagnostic','offset_world_m':offset.tolist(),
               'natural_reconstruction_input':name in ['dvgt_metric','dvgt_lidar_scaled'],
               'extra_information':protocol['extra_information'],'human_verdict':None}
            (case/'input_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
            for f in ['trajectory.npz','initial_rgb.png',* [x.name for x in BASE.glob('reference-*.png')]]:
                (case/f).symlink_to(BASE/f)
            (case/'prompt.txt').write_text('A forward-facing driving camera at an urban intersection in daylight. A white sedan is ahead, with buildings, lane markings, sidewalks, and other vehicles around the road.\n')
            call('scripts/worldsim_v75/render_argoverse.py',case/'render.log',['--run-dir',case])
            if name!='gt_clean':verify_condition(case,target)
            if name=='gt_clean':
                print(json.dumps({'stage':'encode','variant':name}),flush=True)
                call('scripts/worldsim_v75/run_prepared.py',case/'encode.log',['encode','--input-dir',case])
            else:(case/'embeddings.pt').symlink_to(OUT/'gt_clean/embeddings.pt')
            print(json.dumps({'stage':'generate','variant':name}),flush=True)
            call('scripts/worldsim_v75/run_prepared.py',case/'generate.log',['generate','--input-dir',case,'--seed','42'])
            call('scripts/worldsim_v75/evaluate_natural_rollout.py',case/'evaluate.log',['evaluate','--run-dir',OUT,'--case',name])
            result['completed'].append(name)
            (OUT/'queue_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'stage':'complete','variant':name}),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-began
        (OUT/'queue_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
