"""CPU审计实际车辆贡献与固定道路监督域，保留未通过证据。"""
from pathlib import Path
import json,numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
TEACHER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_common1152_20261121/teacher')
rows=[]
for stage in ['canonical_prior_init','joint_fit','handoff_fixed_guarded']:
    for raw in [86,110]:
        with np.load(ROOT/'delivery'/stage/f'{raw:03d}'/'render_float.npz',allow_pickle=False) as z:alpha=z['vehicle_alpha_delete']
        legal=np.asarray(Image.open(ROOT/'delivery/references'/f'{raw:03d}'/'road_legal.png'))>0;bbox=np.asarray(Image.open(ROOT/'delivery/references'/f'{raw:03d}'/'vehicle_potential_bbox.png'))>0
        with np.load(TEACHER/f'raw{raw:03d}'/'supervision.npz',allow_pickle=False) as t:C=t['C'].copy()
        for name,mask in [('C_legal',legal&C),('all_legal',legal)]:
            a=alpha[mask];rows.append(dict(stage=stage,raw=raw,domain=name,domain_pixels=int(mask.sum()),pixels_vehicle_alpha_gt_0=int((a>0).sum()),pixels_vehicle_alpha_gt_1e_3=int((a>1e-3).sum()),pixels_vehicle_alpha_gt_005=int((a>.05).sum()),vehicle_alpha_max=float(a.max()) if a.size else None,vehicle_alpha_sum=float(a.sum()),physical_a20_box_intersection_pixels=int((mask&bbox).sum())))
result=dict(schema='teacher_actual_vehicle_contribution_audit.v1',passed=False,physical_full_box_exclusion_passed=all(x['physical_a20_box_intersection_pixels']==0 for x in rows),actual_rendered_vehicle_contribution_zero_intersection_passed=all(x['pixels_vehicle_alpha_gt_0']==0 for x in rows),rows=rows,source='saved actual native float render packets; fixed saved road legal masks; no training or GPU calls',GPU_calls=0,optimizer_updates=0,interpretation='Physical a20 OBB exclusion does not exclude all inherited large canonical Gaussian footprints. Some rendered a20 contribution remains in generated C road supervision and full legal road domains. Shared differentiable vehicle DC remains in the DELETE training path, so this is a supervision-isolation failure; no new GPU gradient probe was run.',quality_pass=False,original_contract_preserved=True)
p=ROOT/'qa_teacher_render_overlap.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(p)
