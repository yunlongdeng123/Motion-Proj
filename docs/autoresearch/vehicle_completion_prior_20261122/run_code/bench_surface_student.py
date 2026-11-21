"""CPU 固定域复核：旧 reference/指标定义 + 新 student 原始 float render。"""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
import numpy as np
from PIL import Image

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1')
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_road_joint_20261121_r1')
MASK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/repaired_masks_all_r4')
TEACHER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_common1152_20261121/teacher')
FRAMES=(86,110,134,179,198)
TRAIN=(86,110)
STAGES=('native_before','canonical_prior_init','vehicle_fit','joint_fit')
HW=(768,1152)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def frozen_measure(legacy_root):
    code=Path(legacy_root)/'code'
    module_name='_surface_bench_frozen_joint_detail_20261121'
    for name in ('road_detail_metrics',):
        loaded=sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve()!=(code/(name+'.py')).resolve():
            raise ValueError('different metric module already imported: '+name)
    sys.path.insert(0,str(code))
    spec=importlib.util.spec_from_file_location(module_name,code/'joint_detail_metrics.py')
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    spec.loader.exec_module(module)
    return module.measure


def image(path):
    array=np.asarray(Image.open(path).convert('RGB'),np.float32)/255
    if array.shape!=(*HW,3):raise ValueError('frozen reference RGB shape: '+str(path))
    return array


def mask(path):
    array=np.asarray(Image.open(path))>0
    if array.shape!=HW:raise ValueError('frozen reference mask shape: '+str(path))
    return array


def safe_mean(array,domain):
    return float(np.mean(array[domain],dtype=np.float64)) if domain.any() else None


def coverage(isolated,keep,deleted,domain):
    pixels=int(domain.sum())
    row=dict(pixels=pixels)
    for name,alpha in [('isolated_vehicle',isolated),('joint_keep_visible',keep),('joint_delete_visible',deleted)]:
        covered=int(((alpha>=.95)&domain).sum())
        row[name+'_alpha_ge_095_pixels']=covered
        row[name+'_alpha_ge_095_fraction']=covered/pixels if pixels else None
        row[name+'_alpha_mean']=safe_mean(alpha,domain)
    row['isolated_below_095_pixels']=int(((isolated<.95)&domain).sum())
    row['keep_below_095_despite_isolated_ge_095_pixels']=int(((keep<.95)&(isolated>=.95)&domain).sum())
    row['delete_below_095_despite_isolated_ge_095_pixels']=int(((deleted<.95)&(isolated>=.95)&domain).sum())
    return row


def preservation(pred,baseline,target,domain):
    pixels=int(domain.sum())
    delta=np.abs(pred[domain].astype(np.float64)-baseline[domain].astype(np.float64))
    target_error=np.abs(pred[domain].astype(np.float64)-target[domain].astype(np.float64))
    baseline_error=np.abs(baseline[domain].astype(np.float64)-target[domain].astype(np.float64))
    changed=int((delta.max(axis=1)>1/255).sum()) if pixels else 0
    return dict(pixels=pixels,comparison='new stage KEEP float RGB versus new native_before KEEP float RGB',
        max_abs_rgb_change=float(delta.max()) if pixels else None,
        RGB_MAE_change=float(delta.mean()) if pixels else None,
        pixels_gt_1_255=changed,fraction_gt_1_255=changed/pixels if pixels else None,
        channels_gt_1_255=int((delta>1/255).sum()),
        target_fullRGB_MAE=float(target_error.mean()) if pixels else None,
        native_before_target_fullRGB_MAE=float(baseline_error.mean()) if pixels else None,
        target_MAE_is_preservation_training_loss=False)


def read_render(path):
    required=('joint_keep','joint_delete','vehicle_alpha_keep','vehicle_alpha_delete','vehicle_alpha_isolated','road_alpha')
    with np.load(path,allow_pickle=False) as data:
        arrays={k:np.array(data[k],copy=True) for k in required}
    for key,value in arrays.items():
        shape=(3,*HW) if key in ('joint_keep','joint_delete') else HW
        if value.shape!=shape or value.dtype.kind!='f' or not np.isfinite(value).all():
            raise ValueError('invalid original float render '+str(path)+' key='+key)
    arrays['joint_keep']=arrays['joint_keep'].transpose(1,2,0)
    arrays['joint_delete']=arrays['joint_delete'].transpose(1,2,0)
    return arrays


def reference_domains(legacy_root,mask_root,raw):
    ref=Path(legacy_root)/'delivery/references'/f'{raw:03d}'
    labels_path=Path(mask_root)/f'{raw:03d}'/'labels.png'
    labels=np.asarray(Image.open(labels_path).resize((HW[1],HW[0]),Image.Resampling.NEAREST))
    if labels.shape!=HW:raise ValueError('source labels shape')
    score=mask(ref/'vehicle_mask.png');negative=mask(ref/'vehicle_negative_valid.png')
    bbox=mask(ref/'vehicle_potential_bbox.png');old_ignore=mask(ref/'vehicle_occlusion_ignore.png')
    retained=(labels>0)&(labels!=10)&(labels!=21)
    return dict(target=image(ref/'rgb.png'),vehicle_score=score,negative_valid=negative,
        road_legal=mask(ref/'road_legal.png'),labels=labels,retained=retained,
        coverage_domains=dict(observed_visible=score,
            a9_occluded_target_potential=bbox&(labels==10)&~score,
            other_retained_actor_occluded_potential=bbox&retained&~score,
            old_occlusion_ignore=old_ignore&~score,
            potential_unobserved=bbox&~score),
        source=dict(reference_root=str(ref),labels=str(labels_path),
            labels_resampling='source 480x320 instance labels nearest to1152x768; same frozen source, no prediction-dependent mask',
            fixed_masks=['vehicle_mask.png','vehicle_negative_valid.png','road_legal.png'],
            additional_coverage_only_masks=['vehicle_potential_bbox.png','vehicle_occlusion_ignore.png']))


def teacher_target(root,raw):
    path=Path(root)/f'raw{raw:03d}'/'supervision.npz'
    with np.load(path,allow_pickle=False) as data:
        target=np.where(data['A'][None],data['target_A_srgb'],data['target_C_srgb']).transpose(1,2,0)
    if target.shape!=(*HW,3) or target.dtype.kind!='f' or not np.isfinite(target).all():
        raise ValueError('invalid frozen original float A/C teacher')
    return target,path


def historical_rows(legacy_root):
    path=Path(legacy_root)/'delivery/independent_detail_metrics.json'
    source=json.loads(path.read_text(encoding='utf-8'))
    rows=[copy.deepcopy(r) for r in source['rows'] if r['stage']=='handoff_fixed_guarded']
    expected={(raw,'vehicle') for raw in FRAMES}|{(raw,'road') for raw in TRAIN}
    if len(rows)!=7 or {(r['raw'],r['role']) for r in rows}!=expected:
        raise ValueError('expected exactly seven historical handoff_fixed_guarded rows')
    for row in rows:
        row.update(comparison_role='historical_multi_factor_reference',
            isolated_prior_model_gain_attribution=False,
            comparison_note='historical representation/material, handoff and loss/interface differ; multi-factor reference')
    return rows,str(path)


def run(student_root,legacy_root,mask_root,teacher_root):
    student_root=Path(student_root).resolve();delivery=student_root/'delivery'
    # 完整文件检查一次；formal 未完成时直接报告缺项，不等待或轮询。
    required=[delivery/stage/f'{raw:03d}'/'render_float.npz' for stage in STAGES for raw in FRAMES]
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError('formal delivery incomplete:\n'+'\n'.join(missing))
    measure=frozen_measure(legacy_root)
    history,history_path=historical_rows(legacy_root)
    domains={raw:reference_domains(legacy_root,mask_root,raw) for raw in FRAMES}
    teachers={raw:teacher_target(teacher_root,raw) for raw in TRAIN}
    baseline={raw:read_render(delivery/'native_before'/f'{raw:03d}'/'render_float.npz')['joint_keep'] for raw in FRAMES}
    rows=[];details=[];alpha_rows=[];actor_rows=[];render_sources=[]
    for stage in STAGES:
        for raw in FRAMES:
            path=delivery/stage/f'{raw:03d}'/'render_float.npz';a=read_render(path);d=domains[raw]
            target=d['target'];m=d['vehicle_score'];negative=d['negative_valid']
            alpha=a['vehicle_alpha_keep'];predmask=alpha>.5;valid=m|negative;union=(predmask|m)&valid
            visible_coverage=coverage(a['vehicle_alpha_isolated'],alpha,a['vehicle_alpha_delete'],m)
            row,detail=measure(a['joint_keep'],target,m,teacher_role='real_observation')
            row.update(stage=stage,raw=raw,role='vehicle',
                frame_role='BUILD_real_observation' if raw in TRAIN else 'real_observed_diagnostic_not_blind',
                vehicle_alpha_iou=float(((predmask&m)&valid).sum()/max(1,int(union.sum()))),
                vehicle_alpha_iou_full_image_diagnostic=float((predmask&m).sum()/max(1,int((predmask|m).sum()))),
                alpha_inside_mse=safe_mean((alpha-1)**2,m),
                alpha_outside_observable_mse=safe_mean(alpha**2,negative),
                negative_valid_pixels=int(negative.sum()),hidden_GT=False,
                isolated_vehicle_visible_alpha_ge_095_fraction=visible_coverage['isolated_vehicle_alpha_ge_095_fraction'],
                isolated_vehicle_visible_alpha_ge_095_pixels=visible_coverage['isolated_vehicle_alpha_ge_095_pixels'],
                isolated_vehicle_visible_alpha_mean=visible_coverage['isolated_vehicle_alpha_mean'],
                joint_KEEP_visible_alpha_ge_095_fraction=visible_coverage['joint_keep_visible_alpha_ge_095_fraction'],
                joint_KEEP_visible_alpha_ge_095_pixels=visible_coverage['joint_keep_visible_alpha_ge_095_pixels'],
                joint_KEEP_visible_alpha_mean=visible_coverage['joint_keep_visible_alpha_mean'])
            rows.append(row);details.append(dict(stage=stage,raw=raw,role='vehicle',detail=detail))
            for name,domain in d['coverage_domains'].items():
                alpha_rows.append(dict(stage=stage,raw=raw,domain=name,hidden_GT=False,
                    **coverage(a['vehicle_alpha_isolated'],alpha,a['vehicle_alpha_delete'],domain)))
            actor_domains=[('all_retained_other_actors',None,d['retained'])]
            actor_domains += [(f'native{int(label)}_a{int(label)-1}',int(label),d['labels']==label)
                for label in np.unique(d['labels'][d['retained']])]
            for name,label,domain in actor_domains:
                actor_rows.append(dict(stage=stage,raw=raw,domain=name,native_actor_id=label,
                    raw086_a10_native11_focus=(raw==86 and label==11),
                    overlap_a20_fixed_score_pixels=int((domain&m).sum()),
                    **preservation(a['joint_keep'],baseline[raw],target,domain)))
            if raw in TRAIN:
                teacher,teacher_path=teachers[raw];legal=d['road_legal']
                row,detail=measure(a['joint_delete'],teacher,legal,teacher_role='frozen_pseudo_teacher')
                row.update(stage=stage,raw=raw,role='road',frame_role='BUILD_frozen_teacher',
                    road_alpha_below_095=safe_mean((a['road_alpha']<.95).astype(float),legal),
                    road_alpha_below_005=safe_mean((a['road_alpha']<.05).astype(float),legal),
                    teacher_source=str(teacher_path),hidden_GT=False)
                rows.append(row);details.append(dict(stage=stage,raw=raw,role='road',detail=detail))
            render_sources.append(dict(stage=stage,raw=raw,path=str(path),bytes=path.stat().st_size))
            print(f'fixed-bench {stage} raw{raw:03d}',flush=True)
    expected={(s,r,'vehicle') for s in STAGES for r in FRAMES}|{(s,r,'road') for s in STAGES for r in TRAIN}
    if len(rows)!=28 or {(r['stage'],r['raw'],r['role']) for r in rows}!=expected:
        raise RuntimeError('formal fixed detail row contract differs from 28')
    focuses=[r for r in actor_rows if r['raw086_a10_native11_focus']]
    if len(focuses)!=4 or any(r['pixels']<=0 for r in focuses):
        raise RuntimeError('missing fixed raw086 a10/native11 retained domain')
    contract=dict(
        current_candidate_status='diagnostic_rejected_geometry',promotion_allowed=False,
        current_adaptation='400 updates are prior initialization plus visible real-RGB fitting, not complete teacher-to-geometry/silhouette/novel-view distillation',
        future_entry_policy='grayscale component relations, wheel placement, visible silhouette and measured fit before material; protect real visible body geometry and RGB, generate missing regions only',
        score_masks='all RGB/SSIM/HF/edge score domains are frozen old delivery references; no new loss masks or prediction masks select pixels',
        vehicle_target='old references rgb.png, observed sRGB BOX-resized uint8 as frozen historical comparison',
        road_target='original float supervision.npz: where(A,target_A_srgb,target_C_srgb), identical historical independent convention',
        metric_source=str(Path(legacy_root)/'code/joint_detail_metrics.py'),HF_scales=[1,2,4],
        instance_mask_role='automatic observed instance segmentation proxy, not hidden geometry or RGB ground truth',
        diagnostic_role='raw134/179/198 real observed diagnostics, not blind heldout tests',
        prediction='actual unquantized float render_float.npz; no PNG reconstruction',
        alpha_interpretation='isolated alpha reports self coverage; KEEP/DELETE visible contribution includes other-object depth occlusion; source occlusion regions are fixed',
        retained_actor_interpretation='KEEP change against this new native_before; fullRGB target MAE is diagnostic, not a retained-actor training loss',
        historical_comparison='handoff_fixed_guarded seven copied numeric rows; historical multi-factor reference, not isolated causal prior gain')
    fixed=dict(schema='surface_student_fixed_domain_detail.v1',completed=True,rows=rows,
        status='diagnostic_rejected_geometry',promotion_allowed=False,
        historical_reference_rows=history,historical_reference_source=history_path,
        current_rows=28,historical_rows=7,stages=list(STAGES),frames=list(FRAMES),
        resolution=[HW[1],HW[0]],contract=contract,source_masks={str(r):domains[r]['source'] for r in FRAMES})
    qa=dict(schema='surface_student_independent_cpu_qa.v1',completed=True,
        status='diagnostic_rejected_geometry',promotion_allowed=False,
        contract=contract,alpha_coverage_rows=alpha_rows,retained_actor_rows=actor_rows,
        raw086_a10_native11_rows=focuses,render_sources=render_sources,
        output_counts=dict(current_detail_rows=28,historical_reference_rows=7,alpha_region_rows=len(alpha_rows),retained_actor_rows=len(actor_rows)),
        scientific_or_visual_pass=None,human_verdict=None,problem_solved=None,
        gpu_calls=0,training_updates=0,old_run_modified=False)
    write_json(delivery/'fixed_detail_metrics.json',fixed)
    write_json(delivery/'fixed_detail_metrics_full.json',dict(rows=details))
    write_json(student_root/'independent_qa.json',qa)
    print(json.dumps(dict(completed=True,current_rows=28,historical_rows=7,
        fixed_metrics=str(delivery/'fixed_detail_metrics.json'),independent_qa=str(student_root/'independent_qa.json')),indent=2))


def self_check(legacy_root,output=None):
    measure=frozen_measure(legacy_root)
    y,x=np.mgrid[:64,:64];target=np.stack([(x%7)/7,(y%9)/9,((x+y)%11)/11],axis=-1).astype(np.float32)
    domain=np.zeros((64,64),bool);domain[8:56,8:56]=True
    row,_=measure(target.copy(),target,domain,teacher_role='real_observation')
    assert row['pixels']==2304 and row['RGB_MAE']==0 and abs(row['masked_SSIM']-1)<1e-12
    baseline=np.zeros((2,2,3),float)+.5;pred=baseline.copy();pred[0,0,0]+=2/255
    all_domain=np.ones((2,2),bool)
    changes=preservation(pred,baseline,np.zeros_like(pred),all_domain)
    assert changes['pixels']==4 and changes['pixels_gt_1_255']==1
    assert changes['channels_gt_1_255']==1 and abs(changes['RGB_MAE_change']-(2/255)/12)<1e-14
    isolated=np.asarray([[1,.96],[.5,.99]]);keep=np.asarray([[.2,.96],[.3,.99]])
    alpha=coverage(isolated,keep,isolated,all_domain)
    assert alpha['isolated_vehicle_alpha_ge_095_pixels']==3
    assert alpha['joint_keep_visible_alpha_ge_095_pixels']==2
    assert alpha['keep_below_095_despite_isolated_ge_095_pixels']==1
    assert alpha['isolated_below_095_pixels']==1
    result=dict(passed=True,synthetic_only=True,formal_delivery_read=False,
        checks=['frozen metric import and identity RGB/SSIM on fixed domain',
                'float retained-actor threshold/count and MAE analytic check',
                'isolated self coverage versus joint external occlusion separation'],
        gpu_calls=0,training_updates=0)
    if output:
        output=Path(output)
        if output.exists():raise FileExistsError(output)
        write_json(output,result)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student-root',type=Path,default=RUN/'student')
    parser.add_argument('--legacy-root',type=Path,default=OLD)
    parser.add_argument('--mask-root',type=Path,default=MASK)
    parser.add_argument('--teacher-root',type=Path,default=TEACHER)
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--check-output',type=Path)
    args=parser.parse_args()
    if args.self_check:self_check(args.legacy_root,args.check_output)
    else:run(args.student_root,args.legacy_root,args.mask_root,args.teacher_root)


if __name__=='__main__':main()
