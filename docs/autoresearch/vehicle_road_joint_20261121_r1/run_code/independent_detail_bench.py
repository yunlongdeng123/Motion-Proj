"""只读浮点渲染的固定域评价，不依赖各进程日志是否保留旧阶段。"""
from pathlib import Path
import argparse,json,os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import numpy as np
from PIL import Image
from joint_detail_metrics import measure
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
STAGES=['native_before','canonical_init','canonical_prior_init','vehicle_fit','joint_fit','handoff_fixed','handoff_fixed_guarded']
FRAMES=[86,110,134,179,198]
def read(p):return json.loads(p.read_text())
def image(p):return np.asarray(Image.open(p)).astype(np.float32)/255
def mask(p):return np.asarray(Image.open(p))>0
def mean(a,m):return float(a[m].mean()) if m.any() else None
def write(p,x):
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
    q=argparse.ArgumentParser();q.add_argument('--append',action='store_true');args=q.parse_args()
    out=R/'delivery/independent_detail_metrics.json';full=R/'delivery/independent_detail_metrics_full.json'
    rows=read(out)['rows'] if args.append and out.exists() else []
    details=read(full)['rows'] if args.append and full.exists() else []
    # 已验收的25条车辆指标直接复用；道路改用未量化的原教师，新增交接组单独计算。
    if not rows and (R/'delivery/detail_metrics.json').exists():
        rows=[r for r in read(R/'delivery/detail_metrics.json')['rows'] if r['role']=='vehicle']
        if (R/'delivery/detail_metrics_full.json').exists():
            details=[r for r in read(R/'delivery/detail_metrics_full.json')['rows'] if r['role']=='vehicle']
    done={(r['stage'],r['raw'],r['role']) for r in rows}
    for stage in STAGES:
        if not (R/'delivery'/stage).is_dir():continue
        for raw in FRAMES:
            p=R/'delivery'/stage/f'{raw:03d}'/'render_float.npz'
            if not p.exists():raise FileNotFoundError(p)
            with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
            ref=R/'delivery/references'/f'{raw:03d}'
            for role in ['vehicle']+(['road'] if raw in [86,110] else []):
                if (stage,raw,role) in done:continue
                if role=='vehicle':
                    target=image(ref/'rgb.png');m=mask(ref/'vehicle_mask.png');negative=mask(ref/'vehicle_negative_valid.png')
                    pred=a['joint_keep'].transpose(1,2,0);alpha=a['vehicle_alpha_keep'];predmask=alpha>.5;valid=m|negative
                    union=(predmask|m)&valid
                    auxiliary=dict(vehicle_alpha_iou=float(((predmask&m)&valid).sum()/max(1,int(union.sum()))),
                        vehicle_alpha_iou_full_image_diagnostic=float((predmask&m).sum()/max(1,int((predmask|m).sum()))),
                        alpha_inside_mse=mean((alpha-1)**2,m),alpha_outside_observable_mse=mean(alpha**2,negative),
                        negative_valid_pixels=int(negative.sum()),hidden_GT=False)
                else:
                    teacher=R.parent/'road_common1152_20261121/teacher'/f'raw{raw:03d}'/'supervision.npz'
                    with np.load(teacher,allow_pickle=False) as z:
                        target=np.where(z['A'][None],z['target_A_srgb'],z['target_C_srgb']).transpose(1,2,0)
                    m=mask(ref/'road_legal.png');pred=a['joint_delete'].transpose(1,2,0)
                    alpha=a['road_alpha'];auxiliary=dict(road_alpha_below_095=mean((alpha<.95).astype(float),m),
                        road_alpha_below_005=mean((alpha<.05).astype(float),m))
                row,detail=measure(pred,target,m,teacher_role='real_observation' if role=='vehicle' else 'frozen_pseudo_teacher')
                row.update(stage=stage,raw=raw,role=role,**auxiliary)
                rows.append(row);details.append(dict(stage=stage,raw=raw,role=role,detail=detail))
                print(stage,raw,role,flush=True)
                write(out,dict(schema='vehicle_road_fixed_domain_detail.v1',rows=rows,mask_role='automatic observed instance segmentation proxy, not hidden GT',resolution=[1152,768],teacher_frames=[86,110],HF_scales=[1,2,4],reference_quantization='vehicle BOX resize uint8 as training; road original float supervision (A observed, C/S frozen generated); prediction actual float native render',existing_vehicle_metrics_reused=True))
                write(full,dict(rows=details))
    print(json.dumps(dict(complete=True,rows=len(rows),stages=sorted(set(r['stage'] for r in rows)))),flush=True)
if __name__=='__main__':main()
