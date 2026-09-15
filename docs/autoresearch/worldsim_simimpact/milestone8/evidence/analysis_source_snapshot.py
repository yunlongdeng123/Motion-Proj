"""按冻结下游定义汇总完整矩阵；不按几何误差选择案例。"""
import json,shutil,tarfile,time
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from shapely.geometry import Polygon

ROOT=Path('/root/autodl-tmp/runs/worldsim_simimpact')
O=ROOT/'WS-SIM-FF-COHORT-PERCEPTION-01/20260915-r1'
F=ROOT/'WS-SIM-FF-PERCEPTION-01/20260915-r1'
I=ROOT/'WS-SIM-INTERACTION-01/20260915-r1'
E=O/'evidence';E.mkdir(exist_ok=False)
reg=json.loads((O/'registration.json').read_text());meta=json.loads((O/'metadata.json').read_text())
run=json.loads((O/'detection_r1/registration.json').read_text());assert run['completed']
classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
methods=['vggt','omega512','dvgt1','pi3x'];matrix={}
def add(row,root,condition):
    r=dict(row);file=root/f'prediction_{r["condition"]}_{r["index"]:02d}.npz'
    r.update(condition=condition,source_prediction=str(file));key=(r['scene'],condition)
    assert key not in matrix;matrix[key]=r
for row in json.loads((F/'detection_r1/summary.json').read_text()):
    condition=row['condition']
    if condition!='real':
        method,condition=condition.split('_',1);condition=f'{method}_six_{condition}'
    add(row,F/'detection_r1',condition)
for row in json.loads((F/'local_asset_detection_r2/summary.json').read_text()):
    if row['condition'].endswith('_original_twelve'):
        add(row,F/'local_asset_detection_r2',row['condition'].replace('_original_twelve','_fill_missing'))
for row in json.loads((O/'detection_r1/summary.json').read_text()):add(row,O/'detection_r1',row['condition'])
assert len(matrix)==102
def pose(tab,token):
    d=meta[tab][token];p=np.eye(4);p[:3,:3]=Quaternion(d['rotation']).rotation_matrix;p[:3,3]=d['translation'];return p
def matched(row,threshold,metric):return {x['instance'] for x in row['scores'][str(threshold)][metric]['matches']}
summary=[];objects=[];candidates=[]
for scene in reg['scenes']:
    real=matrix[(scene,'real')];gt=real['eligible_gt'];sd=meta['sample_data'][meta['sequences'][scene][0]]
    l2e=pose('calibrated_sensor',sd['calibrated_sensor_token'])
    leads={x['instance'] for x in reg['selection'][scene]['metadata_selection']['leads']}
    baseline=matched(real,.5,'center2m')&matched(real,.5,'bev_iou0p5')
    (E/scene).mkdir()
    for condition in reg['conditions']:
        row=matrix[(scene,condition)];z=np.load(row['source_prediction']);polys=[];preds=[]
        dest=E/scene/f'{condition}_prediction.npz';shutil.copy2(row['source_prediction'],dest)
        for b,s,l in zip(z['boxes'],z['scores'],z['labels']):
            box=Box([b[0],b[1],b[2]+b[5]/2],[b[4],b[3],b[5]],Quaternion(axis=[0,0,1],radians=float(b[6])))
            preds.append((box,float(s),classes[int(l)]));polys.append(Polygon(box.bottom_corners()[:2].T))
        counts={}
        for threshold in [.3,.5]:
            counts[str(threshold)]={metric:{'matched':row['scores'][str(threshold)][metric]['matched'],
                'real_matched':real['scores'][str(threshold)][metric]['matched'],
                'lost_from_qualified_real':len(baseline-matched(row,threshold,metric)),
                'gain_over_real':len(matched(row,threshold,metric)-matched(real,threshold,metric))}
                for metric in ['center2m','bev_iou0p5']}
        summary.append({'scene':scene,'condition':condition,'eligible':len(gt),'real_qualified':len(baseline),'counts':counts})
        for g in gt:
            gb=Box(g['center_lidar'],g['wlh'],Quaternion(g['rotation']));gp=Polygon(gb.bottom_corners()[:2].T)
            center=l2e[:3,:3]@gb.center+l2e[:3,3];rr={'scene':scene,'condition':condition,'instance':g['instance'],'class':g['class'],
                'gt':g,'center_ego':center.tolist(),'metadata_lead':g['instance'] in leads,'baseline_qualified':g['instance'] in baseline,'scores':{}}
            for threshold in [.3,.5]:
                pp=[]
                for (box,score,cls),poly in zip(preds,polys):
                    if cls!=g['class'] or score<threshold:continue
                    err=float(np.linalg.norm(box.center[:2]-gb.center[:2]))
                    pp.append({'center_error_m':err,'score':score,'IoU':float(poly.intersection(gp).area/max(poly.union(gp).area,1e-9)),
                        'center_lidar':box.center.tolist(),'wlh':box.wlh.tolist(),'rotation':box.orientation.elements.tolist(),
                        'same_target_identity_unverified':err>4})
                rr['scores'][str(threshold)]={'center_match':g['instance'] in matched(row,threshold,'center2m'),
                    'IoU_match':g['instance'] in matched(row,threshold,'bev_iou0p5'),
                    'nearest_same_class':min(pp,key=lambda p:p['center_error_m']) if pp else None}
            objects.append(rr)
    for method in methods:
        for g in gt:
            if g['instance'] not in baseline:continue
            rows=[next(r for r in objects if r['scene']==scene and r['instance']==g['instance'] and r['condition']==f'{method}_{v}_fill_missing') for v in ['six','twelve']]
            center_loss=all(not r['scores'][str(t)]['center_match'] for r in rows for t in [.3,.5])
            iou_loss=all(not r['scores'][str(t)]['IoU_match'] for r in rows for t in [.3,.5])
            if not (center_loss or iou_loss):continue
            candidates.append({'scene':scene,'method':method,'instance':g['instance'],'class':g['class'],
                'metadata_lead':g['instance'] in leads,'robust_center_loss':center_loss,'robust_IoU_loss':iou_loss,
                'new_to_detector_log':scene not in ['scene-0004','scene-0061'],
                'condition_rows':rows})
candidates.sort(key=lambda r:(not r['new_to_detector_log'],not r['metadata_lead'],not r['robust_center_loss'],r['scene'],r['instance'],r['method']))
chosen=[];targets=set()
for c in candidates:
    if not c['new_to_detector_log'] or (c['scene'],c['instance']) in targets:continue
    chosen.append({k:v for k,v in c.items() if k!='condition_rows'});targets.add((c['scene'],c['instance']))
    if len(chosen)==2:break
aggregate=[]
for condition in reg['conditions']:
    rows=[r for r in summary if r['condition']==condition]
    aggregate.append({'condition':condition,'eligible':sum(r['eligible'] for r in rows),'real_qualified':sum(r['real_qualified'] for r in rows),
        'counts':{str(t):{m:{k:sum(r['counts'][str(t)][m][k] for r in rows) for k in ['matched','real_matched','lost_from_qualified_real','gain_over_real']} for m in ['center2m','bev_iou0p5']} for t in [.3,.5]}})
decision={'task_id':reg['task_id'],'detector_forwards_this_run':82,'matrix_entries':102,'reused_entries':20,'cumulative_detector_forwards':222,
    'candidate_count':len(candidates),'selected_new_targets':chosen,'scope':'Six exposed logs, one timestamp each. Current scan isolation with real intensity/history; no complete simulator or independent test claim.',
    'failure_ledger_delta':'pending','human_verdict':None,'goal_status':'active','analysis_time_unix':time.time()}
for name,data in [('matrix_rows',list(matrix.values())),('scene_counts',summary),('objects',objects),('aggregate',aggregate),('candidates',candidates),('decision',decision),('registration',reg)]:
    (E/f'{name}.json').write_text(json.dumps(data,indent=2))
for src,name in [(O/'detection_r1/registration.json','detector_registration.json'),(O/'extraction_result.json','extraction_result.json'),(O/'metadata.json','metadata.json')]:shutil.copy2(src,E/name)
shutil.copy2(__file__,E/'analysis_source_snapshot.py')
with tarfile.open(O/'evidence.tar.gz','w:gz') as t:
    for p in E.rglob('*'):
        if p.is_file():t.add(p,arcname=p.relative_to(E))
print('COMPLETE_MATRIX',len(matrix),'eligible',aggregate[0]['eligible'],'real qualified',aggregate[0]['real_qualified'],flush=True)
for r in aggregate:print(r['condition'],{t:(r['counts'][t]['center2m']['matched'],r['counts'][t]['bev_iou0p5']['matched']) for t in ['0.3','0.5']},flush=True)
print('CANDIDATES',len(candidates),'SELECTED',chosen,flush=True)
