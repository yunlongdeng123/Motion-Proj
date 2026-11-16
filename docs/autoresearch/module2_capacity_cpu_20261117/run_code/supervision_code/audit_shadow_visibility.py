"""CPU-only read of cached S2 road coverage and same-sort branch alphaT."""
from pathlib import Path
import argparse,hashlib,json,os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def group_statistics(mask,alpha,road,static,dynamic):
    coverage=[mask&(alpha<.05),mask&(alpha>=.05)&(alpha<=.9),mask&(alpha>.9)]
    contribution=[mask&(road==0),mask&(road>0)&(road<=.05),mask&(road>.05)]
    if sum(int(x.sum()) for x in coverage)!=int(mask.sum()) or sum(int(x.sum()) for x in contribution)!=int(mask.sum()):raise ValueError('bins changed a fixed denominator')
    return dict(pixels=int(mask.sum()),road_only_coverage=dict(gap_lt_point05=int(coverage[0].sum()),thin_point05_to_point9=int(coverage[1].sum()),support_gt_point9=int(coverage[2].sum())),
        joint_road_alphaT=dict(exact_zero=int(contribution[0].sum()),positive_le_point05=int(contribution[1].sum()),gt_point05=int(contribution[2].sum()),
            mean=float(road[mask].mean()) if mask.any() else None,quantiles=np.quantile(road[mask],[0,.1,.5,.9,1]).tolist() if mask.any() else None),
        coverage_by_joint_contribution_matrix=[[int((a&b).sum()) for b in contribution] for a in coverage],
        static_alphaT_dominant_pixels=int((mask&(static>=road)&(static>=dynamic)).sum()),
        dynamic_alphaT_dominant_pixels=int((mask&(dynamic>static)&(dynamic>=road)).sum()),
        joint_road_alphaT_exceeds_road_only_alpha_gt_2e_6=int((mask&(road>alpha+2e-6)).sum()))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only audit requires CUDA_VISIBLE_DEVICES empty')
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rendered',type=Path,default=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/rendered'))
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);completed=read(args.rendered/'completed.json')
    if not completed.get('completed') or not completed.get('same_depth_sort'):raise ValueError('cached source must declare completed same-sort public rendering')
    rows=[];sources=[]
    for raw in [86,110]:
        s=load(args.candidate/f'{raw:03d}.npz');path=args.rendered/f'{raw:03d}'/'arrays.npz';a=load(path)
        fields=['road_only_alpha','road_alphaT','static_alphaT','dynamic_alphaT','total_alpha']
        if any(a[k].shape!=s['K'].shape or not np.isfinite(a[k]).all() for k in fields):raise ValueError('cached branch arrays have invalid shape or values')
        alpha,road,static,dynamic=[a[k] for k in fields[:4]]
        exact_reference=np.array_equal(alpha,s['fixed_initial_alpha'])
        groups={name:group_statistics(mask,alpha,road,static,dynamic) for name,mask in [
            ('full_S_target_shadow',s['S_target_shadow']),('photo_S_target_shadow',s['S_target_shadow_photo']),
            ('S_excluded_low_reference_alpha',s['S_target_shadow']&~s['S_target_shadow_photo'])]}
        rows.append(dict(raw=raw,cached_public_alpha_matches_frozen_r3_reference_exact=exact_reference,groups=groups,
            joint_alpha_branch_conservation_max=float(np.abs(static+dynamic+road-a['total_alpha']).max())))
        sources.extend(dict(path=str(p),sha256=sha(p)) for p in [args.candidate/f'{raw:03d}.npz',path])
        np.savez_compressed(args.output/f'{raw:03d}_visibility_evidence.npz',full_S=s['S_target_shadow'],photo_S=s['S_target_shadow_photo'],
            road_only_alpha=alpha,road_alphaT=road,static_alphaT=static,dynamic_alphaT=dynamic)
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    sources.append(dict(path=str(args.rendered/'completed.json'),sha256=sha(args.rendered/'completed.json')))
    write(args.output/'shadow_coverage_and_visibility.json',dict(schema='road_capacity_shadow_coverage_visibility.v1',rows=rows,source_files=sources,
        coverage_thresholds=dict(gap='alpha<.05',thin='.05<=alpha<=.9',support='alpha>.9'),
        contribution_thresholds=dict(zero='exact alphaT==0',low='0<alphaT<=.05',above_low='alphaT>.05, numeric contribution bin only; not a perceptual quality pass'),
        matrix_rows=['road_only_gap','road_only_thin','road_only_support'],matrix_columns=['joint_road_exact_zero','joint_road_positive_le_point05','joint_road_gt_point05'],
        RGB_divided_by_alpha=False,small_alpha_division=False,evaluation_masks_changed=False,
        joint_dominance_is_not_wrong_ownership_proof=True,cached_source_only=True,new_GPU_operations=0,new_training_updates=0,new_render_operations=0,
        future_GPU_preflight=['read-only baseline and absolute-allowed max opacity/tangent-sigma coverage states on fixed full S',
            'classify controllable thin support versus geometry gaps and native static/dynamic occlusion',
            'derive actual S contributor scope and verify retained joint K before any color-only comparison'],
        conclusion='color-only fitting uses fixed high-coverage S photo, while the full S denominator and low-coverage exclusions remain explicit; no larger generator explains opacity or joint occlusion failures'))


if __name__=='__main__':main()
