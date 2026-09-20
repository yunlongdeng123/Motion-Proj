"""原生规划评价参考单独落盘，保持 actor 真值不进入模型输入。"""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,default=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1'));ap.add_argument('--input-dir',type=Path,default=R/'real_inputs_r2');ap.add_argument('--out-dir',type=Path,default=R/'evaluation_r1');args=ap.parse_args()
Q=args.source_root;O=args.out_dir;O.mkdir(exist_ok=False);src=args.input_dir;input_reg=json.loads((src/'registration.json').read_text());scene=input_reg['scene'];indices=input_reg['evaluation_original_indices']
meta=json.loads((Q/'metadata'/f'{scene}.json').read_text());seq=sorted(meta['sample'].values(),key=lambda s:s['timestamp'])
inputs=list({x['original_index']:x for x in json.loads((src/'policy_inputs.json').read_text())}.values());refs={r['original_index']:r for r in json.loads((src/'evaluation_references.json').read_text())}
tokens={s['token'] for s in seq[min(indices)+1:max(indices)+7]};by_sample={}
for a in json.loads((Q/'data/v1.0-trainval/sample_annotation.json').read_text()):
    if a['sample_token'] in tokens and a['num_lidar_pts']>0:by_sample.setdefault(a['sample_token'],[]).append(a)
rows=[]
for x in inputs:
    i=x['original_index']
    if i not in indices:continue
    inv=np.linalg.inv(np.array(x['lidar2global']));future=[];instances=[]
    for s in seq[i+1:i+7]:
        boxes=[];ids=[]
        for a in by_sample.get(s['token'],[]):
            center=inv[:3,:3]@np.array(a['translation'])+inv[:3,3];rotation=inv[:3,:3]@Quaternion(a['rotation']).rotation_matrix
            w,l,h=a['size'];boxes.append([*center,l,w,h,float(np.arctan2(rotation[1,0],rotation[0,0]))]);ids.append(a['instance_token'])
        future.append(boxes);instances.append(ids)
    rows.append({**refs[i],'future_boxes_lidar_lwh':future,'future_box_instances':instances})
assert len(rows)==8 and all(len(r['future_boxes_lidar_lwh'])==6 for r in rows)
(O/'references.json').write_text(json.dumps(rows,indent=2));shutil.copy2(__file__,O/'source_snapshot.py')
registration={'scene':scene,'source_inputs':str(src),'prepared_unix':time.time(),'scope':'Official SparseDrive PlanningMetric on all eight registered starts; native six waypoints /3s. Future annotation boxes with >0 LiDAR points, as dataset.get_ann_info; l,w,h and yaw transformed into current LiDAR frame.',
 'reference_origin':'Future LiDAR sensor origin positions, matching official converter. Not interchangeable with rear axle or actor center.',
 'metrics':'Preserve native L2 prefix means and collision flags at every horizon; also report raw endpoint L2. Native footprint4.084x1.85m with0.5m offset; GT-overlap-masked model collisions and unmasked/GT flags separately.',
 'limits':'Open-loop native-dataset diagnostic, not a sensor feedback loop, severity/at-fault metric, full road fidelity or independent held-out benchmark.',
 'frozen_decision':'No reconstruction claim from baseline. Continue to asset/render comparisons only after adequate real-input performance is established, without tuning routes or choosing a subset of these eight starts.',
 'expected_evaluated_starts':8,'human_verdict':None}
(O/'registration.json').write_text(json.dumps(registration,indent=2));print('EVALUATOR_PREPARED',len(rows),flush=True)
