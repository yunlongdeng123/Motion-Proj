from pathlib import Path
import os,json
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import numpy as np
from PIL import Image
from joint_detail_metrics import measure
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
STAGES=['native_before','canonical_init','canonical_prior_init','vehicle_fit','joint_fit']
FRAMES=[86,110,134,179,198]
metrics=json.loads((ROOT/'delivery/metrics.json').read_text())['rows'];index={(x['stage'],x['raw']):x for x in metrics};rows=[];details=[]
for stage in STAGES:
    for raw in FRAMES:
        ref=ROOT/'delivery/references'/f'{raw:03d}'
        with np.load(ROOT/'delivery'/stage/f'{raw:03d}'/'render_float.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        target=np.asarray(Image.open(ref/'rgb.png')).astype(np.float32)/255;mask=np.asarray(Image.open(ref/'vehicle_mask.png'))>0
        row,detail=measure(a['joint_keep'].transpose(1,2,0),target,mask,teacher_role='real_observation')
        old=index[(stage,raw)];row.update(stage=stage,raw=raw,role='vehicle',vehicle_alpha_iou=old['vehicle_alpha_iou_valid'],vehicle_alpha_iou_full_image_diagnostic=old['vehicle_alpha_iou_full_image_diagnostic'],alpha_inside_mse=old['vehicle_alpha_inside_mse'],alpha_outside_observable_mse=old['vehicle_alpha_outside_observable_mse'],hidden_GT=False)
        rows.append(row);details.append(dict(stage=stage,raw=raw,role='vehicle',detail=detail))
        if raw in (86,110):
            target=np.asarray(Image.open(ref/'road_teacher.png')).astype(np.float32)/255;mask=np.asarray(Image.open(ref/'road_legal.png'))>0
            row,detail=measure(a['joint_delete'].transpose(1,2,0),target,mask,teacher_role='frozen_pseudo_teacher')
            row.update(stage=stage,raw=raw,role='road',road_alpha_below_095=old['road_alpha_below_095'],road_alpha_below_05=old['road_alpha_below_05'],road_black_fraction=old['road_black_fraction'])
            rows.append(row);details.append(dict(stage=stage,raw=raw,role='road',detail=detail))
        print(stage,raw,flush=True)
(ROOT/'delivery/detail_metrics.json').write_text(json.dumps(dict(rows=rows,stages=STAGES,vehicle_mask='fixed real accepted repaired visible body segmentation proxy; not manually labeled hidden GT',road_mask='fixed legal A/C/S & actual base alpha>.05, excludes target full projected box and retained actors',reference_quantization='references PNG represent original resize BOX and frozen teacher 8-bit appearance; prediction uses actual float NPZ'),indent=2,allow_nan=False))
(ROOT/'delivery/detail_metrics_full.json').write_text(json.dumps(dict(rows=details),indent=2,allow_nan=False))
