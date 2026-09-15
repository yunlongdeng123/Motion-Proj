"""导出两次有限基线实验的全部证据；不重跑模型。"""
import json,shutil,tarfile
from pathlib import Path
from PIL import Image
Q=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1')
O=Q/'evidence';O.mkdir(exist_ok=True)
for f in ['registration.json','selection_summary.json','raw_source_availability.json','extraction_result.json','fit_admission.json','fit_admission_route.json']:
    shutil.copy2(Q/f,O/f)
for condition in ['baseline','baseline_route_r1']:
    root=Q/condition;out=O/condition;out.mkdir(exist_ok=True)
    shutil.copy2(root/'registration.json',out/'registration.json')
    for f in ['policy_probe_summary.json','pdm_simulation_summary.json']:shutil.copy2(root/'lidar_policy'/f,out/f)
    for p in (root/'lidar_policy/pdm_simulation').glob('*/states.npz'):
        dest=out/p.parent.name;dest.mkdir(exist_ok=True);shutil.copy2(p,dest/'states.npz')
    shutil.copy2(root/'lidar_policy/scene-0002_start00_log_reference.json',out/'start00_reference.json')
inp=json.loads((Q/'baseline/inputs/scene-0002_start00.json').read_text())
view=next(v for v in inp['views'] if v['camera']=='CAM_FRONT' and v['sample_index']==0)
Image.open(view['image']).save(O/'scene0002_start00_front.jpg',quality=94)
(O/'image_source.json').write_text(json.dumps(view,indent=2))
shutil.copy2(__file__,O/'collection_source_snapshot.py')
with tarfile.open(Q/'fresh_baseline_evidence.tar.gz','w:gz') as tar:tar.add(O,arcname='evidence')
print('EVIDENCE_SAVED',str(Q/'fresh_baseline_evidence.tar.gz'),flush=True)
