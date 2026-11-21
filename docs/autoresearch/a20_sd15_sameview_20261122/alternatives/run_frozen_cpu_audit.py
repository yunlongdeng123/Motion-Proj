"""Run the unchanged HQ CPU gate once per already frozen alternative teacher."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
SOURCE=CAMPAIGN/'audit/source/audit_hq_cpu.py'
LEGACY=CAMPAIGN/'audit/source/shape_audit_cpu.py'
GATE=CAMPAIGN/'audit/preregistered_gate.json'
CONDITIONING=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1/conditioning')


def record(path):
    p=Path(path)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['trellis','amodal3r'],required=True)
    args=parser.parse_args()
    name='trellis_hq_rear_seed101_r1' if args.model=='trellis' else 'amodal_real_raw110_seed101_r2'
    mesh=ROOT/'runs'/name/'mesh.ply'
    target=ROOT/'cpu_audits'/name
    target.mkdir(parents=True,exist_ok=False)
    forward='-y' if args.model=='trellis' else '+y'
    rotation=[[0,-1,0],[1,0,0],[0,0,1]] if args.model=='trellis' else [[0,1,0],[-1,0,0],[0,0,1]]
    plan={'schema':'a20.alternative_frozen_cpu_audit.v1','model':args.model,
          'candidate':name,'mesh':record(mesh),'approved_proper_rotation_3x3':rotation,
          'raw_axis_forward':forward,'raw_axis_up':'+z','uniform_length_m':5.0,
          'anisotropic_scaling':False,'source':record(SOURCE),'legacy':record(LEGACY),
          'gate':record(GATE),'point_bank':record(CONDITIONING/'lidar/aggregated_lidar_train_086_110.npz'),
          'conditioning':str(CONDITIONING),'point_count':1030,'evaluated_raws':[86,110],
          'heldout_used':False,'gpu_used':False,'max_icp_iterations':12,
          'method_or_threshold_changes':False,'stages':[],
          'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'state':'running'}
    def save():
        (target/'execution_receipt.json').write_text(json.dumps(plan,indent=2)+'\n')
    (target/'frozen_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    save()
    shared=['--conditioning',str(CONDITIONING),'--legacy',str(LEGACY)]
    commands=[
        ('initialize',['initialize','--mesh',str(mesh),'--output',str(target/'uniform'),
                       '--forward='+forward,'--up=+z','--semantic-reason',
                       'Root independently inspected raw front/rear/top/bottom images and approved this exact proper rotation.']),
        ('audit_uniform',['audit','--mesh',str(target/'uniform/candidate_actor_uniform_init.ply'),
                          '--output',str(target/'uniform_audit'),*shared,'--gate',str(GATE),'--label',name+'_uniform']),
        ('register_once',['register','--mesh',str(target/'uniform/candidate_actor_uniform_init.ply'),
                          '--output',str(target/'registered'),*shared]),
        ('audit_registered',['audit','--mesh',str(target/'registered/candidate_actor_bounded_registered.ply'),
                             '--output',str(target/'registered_audit'),*shared,'--gate',str(GATE),'--label',name+'_bounded_registered']),
    ]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    started=time.time()
    for stage,arguments in commands:
        print('CPU_STAGE_START',args.model,stage,flush=True)
        step=time.time();command=[sys.executable,str(SOURCE),*arguments]
        with (target/(stage+'.log')).open('w') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        item={'stage':stage,'command':command,'returncode':proc.returncode,'seconds':time.time()-step}
        plan['stages'].append(item);save()
        print('CPU_STAGE_FINISH',args.model,stage,proc.returncode,round(item['seconds'],3),flush=True)
        if proc.returncode:
            plan.update(state='failed',error_stage=stage);save();return proc.returncode
        if stage=='initialize':
            actual=json.loads((target/'uniform/semantic_uniform_transform.json').read_text())
            if actual['proper_rotation_3x3']!=rotation:
                raise RuntimeError('Semantic proper rotation differs from root approval.')
    plan.update(state='complete',seconds=time.time()-started,
                completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save()
    for stage in ['uniform_audit','registered_audit']:
        result=json.loads((target/stage/'audit.json').read_text())
        print(json.dumps({'stage':stage,'numeric_failures':result['numeric_failures'],
                          'point_to_surface_m':result['point_to_surface_m'],
                          'extent_to_known_size':result['extent_to_known_size'],
                          'projection_metrics':[{k:p[k] for k in ['raw','visible_core_recall','reliable_domain_iou','known_background_spill_fraction']} for p in result['projection_metrics']]}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
