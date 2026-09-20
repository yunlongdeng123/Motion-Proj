"""只改变官方随机seed的有限四组复核，复用原条件与编码，不重做筛查。"""
import json
import os
import shutil
import time
from datetime import datetime,timezone
from pathlib import Path
from run_natural_rollouts import OUT as SOURCE, call

OUT=SOURCE.parent/'20260920-seed43'
VARIANTS=['gt_clean','dvgt_metric','ordinary_bbox','reference_lidar']

def main():
    assert json.loads((SOURCE/'queue_result.json').read_text())['status']=='complete'
    assert json.loads((SOURCE/'review_result.json').read_text())['status']=='complete'
    assert not OUT.exists(),'拒绝覆盖或自动重试已启动复核'
    OUT.mkdir(parents=True)
    p=json.loads((SOURCE/'protocol.json').read_text())
    p.update(run_id='20260920-seed43',seed=43,variants=VARIANTS,
             frozen_utc=datetime.now(timezone.utc).isoformat(),source_run=str(SOURCE),
             selection='fixed replication of the same discovery scene and target; no new source or amplitude selection',
             comparison_plan={'primary':'mean 2D distance to recorded real RGB on unchanged frames [0,15,30,45,60]',
                              'repair':'DVGT minus extra-target-LiDAR mean; also report each of four noninitial times',
                              'ordinary_control':'DVGT minus ordinary-bbox mean and per-time direction, retaining larger 3D error of this control',
                              'gt_control':'DVGT minus GT-condition mean and per-time direction, not only repair contrast',
                              'boundary':'two seeds of one scene are not independent scenes; frame2s includes pedestrian occlusion',
                              'stop':'no additional seeds or offset changes in this replication; if recovery does not repeat, close this candidate'},
             human_verdict=None)
    (OUT/'protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    for f in ['evaluator_reference.json',*[x.name for x in SOURCE.glob('reference-*.png')]]:
        (OUT/f).symlink_to(SOURCE/f)
    rows=[];result={'status':'running','pid':os.getpid(),'seed':43,'completed':rows,'human_verdict':None}
    (OUT/'queue_result.json').write_text(json.dumps(result,indent=2)+'\n')
    began=time.monotonic()
    try:
        for name in VARIANTS:
            case=OUT/name;case.mkdir();source=SOURCE/name
            for file in ['conditions.npy','embeddings.pt','initial_rgb.png','prompt.txt','scene.json','trajectory.npz','render_result.json']:
                assert (source/file).exists(),source/file
                (case/file).symlink_to((source/file).resolve())
            binding=source/'condition_binding_check.json'
            if binding.exists():(case/binding.name).symlink_to(binding)
            manifest=json.loads((source/'input_manifest.json').read_text())
            manifest.update(run_id='20260920-seed43',seed=43,source_run=str(source),
                            role='same_scene_fixed_seed_replication',human_verdict=None)
            (case/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
            for file in ['conditions.npy','embeddings.pt','initial_rgb.png','prompt.txt']:
                assert (case/file).resolve()==(source/file).resolve()
            print(json.dumps({'stage':'generate','variant':name,'seed':43}),flush=True)
            call('scripts/worldsim_v75/run_prepared.py',case/'generate.log',['generate','--input-dir',case,'--seed','43'])
            call('scripts/worldsim_v75/evaluate_natural_rollout.py',case/'evaluate.log',['evaluate','--run-dir',OUT,'--case',name])
            rows.append(name);(OUT/'queue_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'stage':'complete','variant':name,'seed':43}),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-began
        (OUT/'queue_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
