"""冻结匹配判据之外保留连续定位误差，防止把2m阈值跨越误写为目标消失。"""
import json,tarfile,shutil
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from shapely.geometry import Polygon
F=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FF-PERCEPTION-01/20260915-r1')
S=F/'local_asset_inputs_r2';R=F/'local_asset_detection_r2';O=F/'local_asset_evidence_r2'
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir();shutil.copy2(__file__,O/'summary_source_snapshot.py')
reg=json.loads((R/'registration.json').read_text());assert reg['completed']
supports=json.loads((F/'candidate_support_r1/summary.json').read_text());original=json.loads((F/'detection_r1/summary.json').read_text());new=json.loads((R/'summary.json').read_text());out=[]
classes=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
for s in supports:
    method=s['method'];g=s['gt'];gb=Box(g['center_lidar'],g['wlh'],Quaternion(g['rotation']));gp=Polygon(gb.bottom_corners()[:2].T)
    for variant in ['six','twelve']:
        for condition in ['original','local_radial','same_faces_deleted']:
            if condition=='original' and variant=='six':
                row=next(r for r in original if r['scene']=='scene-0004' and r['condition']==method+'_fill_missing');root=F/'detection_r1'
            else:
                suffix='original_twelve' if condition=='original' else condition
                row=next(r for r in new if r['condition']==method+'_'+variant+'_'+suffix);root=R
            file=root/f'prediction_{row["condition"]}_{row["index"]:02d}.npz';z=np.load(file);matches={}
            for threshold in [.3,.5]:
                cand=[]
                for b,score,label in zip(z['boxes'],z['scores'],z['labels']):
                    if classes[int(label)]!=s['class'] or score<threshold:continue
                    box=Box([b[0],b[1],b[2]+b[5]/2],[b[4],b[3],b[5]],Quaternion(axis=[0,0,1],radians=float(b[6])));pp=Polygon(box.bottom_corners()[:2].T)
                    cand.append({'center_error_m':float(np.linalg.norm(box.center[:2]-gb.center[:2])),'score':float(score),
                        'BEV_IoU':float(pp.intersection(gp).area/pp.union(gp).area),'center_lidar':box.center.tolist(),'wlh':box.wlh.tolist(),'rotation':box.orientation.elements.tolist()})
                best=min(cand,key=lambda p:p['center_error_m']) if cand else None
                matches[str(threshold)]={'nearest_same_class_prediction':best,
                    'center_match':any(m['instance']==s['instance'] for m in row['scores'][str(threshold)]['center2m']['matches']),
                    'IoU_match':any(m['instance']==s['instance'] for m in row['scores'][str(threshold)]['bev_iou0p5']['matches'])}
            out.append({'method':method,'variant':variant,'condition':condition,'instance':s['instance'],'class':s['class'],'scores':matches,'source_prediction':str(file)})
            shutil.copy2(file,O/(method+'_'+variant+'_'+condition+'_prediction.npz'))
(O/'target_results.json').write_text(json.dumps(out,indent=2))
for root,name in [(F/'candidate_support_r1','support'),(S,'inputs'),(R,'detection')]:
    dest=O/name;dest.mkdir()
    for fn in ['registration.json','summary.json','geometry_audit.json','input_manifest.json','source_snapshot.py']:
        if (root/fn).exists():shutil.copy2(root/fn,dest/fn)
for p in (F/'local_asset_context_r2').iterdir():shutil.copy2(p,O/p.name)
decision={'task_id':'WS-SIM-FF-LOCAL-ASSET-01','completed_detector_forwards':10,'cumulative_detector_forwards':140,
          'Omega':'Local mesh radial correction improves target readout and detection localization in six/twelve view assets, but same-face deletion also recovers; no evidence that generative repair is necessary. A local causal effect within this fixed-history sensor adapter is supported.',
          'Pi3X':'Radial range error decreases; six-view detection does not recover. Twelve-view original already has a center match, which weakens after repair; no stable improvement.',
          'admission':'Does not pass the preregistered beyond-deletion Hero criterion. Single exposed train frame, no temporal or independent scene confirmation, no planning/safety result.',
          'engineering':'First array preparation grouped finite and restored rays. Corrected to preserve original ray order before detector execution; only r2 inputs were inferred. Geometry protocol unchanged.',
          'failure_ledger_delta':'V74-H2-F20','human_verdict':None,'goal_status':'active'}
(O/'decision.json').write_text(json.dumps(decision,indent=2))
with tarfile.open(F/'local_asset_evidence_r2.tar.gz','w:gz') as t:
    for p in O.rglob('*'):
        if p.is_file():t.add(p,arcname=str(p.relative_to(O)))
for r in out:
    p=r['scores']['0.3']['nearest_same_class_prediction'];print(r['method'],r['variant'],r['condition'],p and (p['center_error_m'],p['score'],p['BEV_IoU']),flush=True)
