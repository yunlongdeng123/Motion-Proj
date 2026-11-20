"""冻结真实白漆ROI内的有限外观边缘检验；0GPU/0native/0优化。"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time,gc
import numpy as np
from scipy import ndimage as ndi

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1')
ROOT=B/'marking_reference';OUT=ROOT/'local_appearance_test_r1';RAW=86
sys.path.insert(0,str(B/'metrics'))
from road_detail_metrics import DetailConfig,_gradient_edges,_edge_stats,_masked_hf,_hf_stats,_strict_core

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    begin=time.time();OUT.mkdir(parents=True,exist_ok=False)
    registration=json.loads((ROOT/'source_only_r1/086_candidates.json').read_text())
    review=json.loads((ROOT/'source_visual_review.json').read_text());assert review['raw86']['verified_local_paint_candidates']
    selected={c['component_id']:c for c in review['raw86']['verified_local_paint_candidates']}
    source=[c for c in registration['components']if c['component_id']in selected]
    if len(source)!=3:raise RuntimeError('Frozen source-only ROI set changed')
    target=B/'highres/targets/086.npz';alpha_source=B/'highres/rendered/M11/086.npz'
    with np.load(target,allow_pickle=False)as z:
        T=z['T_A'].transpose(1,2,0).astype(np.float64);A=z['A_pre_alpha'];hit=z['surface_hit'];K=z['K_vehicle']
    with np.load(alpha_source,allow_pickle=False)as z:alpha=z['road_alpha'].astype(np.float64)
    context=A&hit&(alpha>=.95)&~K
    config=DetailConfig();scores={}
    for c in source:
        key=str(c['component_id']);score=np.zeros(context.shape,bool);x0,y0,x1,y1=c['bbox_xyxy'];score[y0:y1,x0:x1]=True;scores[key]=score
        if c['bbox_xyxy']!=selected[c['component_id']]['bbox_xyxy']:raise RuntimeError('ROI identity changed')
    write(OUT/'contract.json',dict(schema='verified_real_A_paint_ROI_appearance_test.v1',UTC=datetime.now(timezone.utc).isoformat(),
        source_only_ROIs_frozen_before_prediction_RGB_read=True,ROIs=[dict(component_id=c['component_id'],bbox_xyxy=c['bbox_xyxy'])for c in source],
        score='Frozen source-only component bounding RECTANGLE, not brightness pixels or a paint-boundary mask',
        context='Whole-image common A_pre_alpha & surface_hit & fixed road_alpha>=.95 & ~K_vehicle; never white-only context',
        methods=['M01','M11','M10'],reference='Actual T_A source appearance edges inside verified visible white-paint local ROI',
        full_lane_GT=False,paint_outer_boundary_GT=False,world_width_GT=False,optimizer_updates=0,native_calls=0))
    ts,ta,tn,edge_context=_gradient_edges(T,context)
    reference_HF={sigma:_masked_hf(T,context,sigma,config.gaussian_truncate)[0]for sigma in config.pixel_scales}
    roi_info=[]
    for c in source:
        key=str(c['component_id']);score=scores[key];scored=score&context;edgecore=scored&edge_context
        item=dict(component_id=c['component_id'],bbox_xyxy=c['bbox_xyxy'],ROI_pixels=int(score.sum()),context_scored_pixels=int(scored.sum()),
            outside_context_pixels=int((score&~context).sum()),edge_core_r2_pixels=int(edgecore.sum()),
            HF_core_pixels={str(s):int((scored&_strict_core(context,int(config.gaussian_truncate*s+.5))).sum())for s in config.pixel_scales},
            source_reference_edges=[dict(threshold=float(t),edges=int((tn&(ts>=t)&edgecore).sum()))for t in config.edge_thresholds])
        roi_info.append(item)
    # Preserve qualification counts before any method predicted RGB is read.
    write(OUT/'source_reference_counts.json',dict(raw=RAW,ROIs=roi_info,target_SHA256=sha(target),fixed_alpha_source_SHA256=sha(alpha_source),
        reference_definition='RGB source-gradient NMS appearance edges within verified paint ROI, not connected-component boundaries',prediction_RGB_read_yet=False))
    rows=[];bindings={str(target):sha(target),str(alpha_source):sha(alpha_source),str(ROOT/'source_visual_review.json'):sha(ROOT/'source_visual_review.json'),str(B/'metrics/road_detail_metrics.py'):sha(B/'metrics/road_detail_metrics.py'),str(Path(__file__)):sha(__file__)}
    for method in ('M01','M11','M10'):
        packet=B/'highres/rendered'/method/'086.npz'
        with np.load(packet,allow_pickle=False)as z:
            actual_alpha=z['road_alpha'].astype(np.float64)
            if not np.array_equal(alpha,actual_alpha):raise RuntimeError('Method alpha differs; no mask intersection allowed')
            P=z['road_rgb'].transpose(1,2,0).astype(np.float64)
        bindings[str(packet)]=sha(packet)
        R=np.divide(P,alpha[...,None],out=np.zeros_like(P),where=(alpha>=.95)[...,None]);del P
        if not np.isfinite(R[context]).all():raise RuntimeError('Nonfinite material RGB')
        ps,pa,pn,_=_gradient_edges(R,context)
        method_HF={sigma:_masked_hf(R,context,sigma,config.gaussian_truncate)[0]for sigma in config.pixel_scales}
        for info in roi_info:
            key=str(info['component_id']);score=scores[key];scored=score&context;edgecore=scored&edge_context
            scales=[]
            for sigma in config.pixel_scales:
                core=scored&_strict_core(context,int(config.gaussian_truncate*sigma+.5))
                scales.append(dict(sigma_pixels=float(sigma),core_pixels=int(core.sum()),boundary_pixels=int((scored&~core).sum()),
                    metrics=_hf_stats(method_HF[sigma],reference_HF[sigma],core)))
            edges=[]
            for threshold in config.edge_thresholds:
                ref=tn&(ts>=threshold)&edgecore;pred=pn&(ps>=threshold)&edgecore
                edges.append(dict(threshold=float(threshold),**_edge_stats(pred,ref,pa,ta,config.edge_tolerances_pixels,config.edge_angle_tolerances_degrees)))
            rows.append(dict(raw=RAW,method=method,component_id=info['component_id'],bbox_xyxy=info['bbox_xyxy'],source_qualification=info,
                color_MAE_auxiliary=float(np.abs(R-T)[scored].mean())if scored.any()else None,
                scales=scales,edge_curves=edges,role='real_observation_appearance_edges_inside_source_verified_paint_ROI',
                lane_geometry_GT=False,paint_width_GT=False,whole_line_continuity_claim=False))
        del R,method_HF,ps,pa,pn;gc.collect()
        print('LOCAL_APPEARANCE_DONE',method,flush=True)
    result=dict(schema='verified_real_A_paint_ROI_appearance_results.v1',completed=True,raw=RAW,resolution=[1920,1280],
        source_reference_counts=roi_info,rows=rows,source_bindings=bindings,seconds=time.time()-begin,
        source_real_paint_identity_verified=True,reference_is_actual_source_gradient_not_brightness_boundary=True,
        source_role='real_observation',no_aggregate_quality_score=True,marking_geometry_GT=False,
        limitation='Only local observed appearance/edge correspondence. These short partially supported source ROIs do not provide full-lane width/centreline/continuity or hidden-road geometry GT. Empty cores remain N/A.',
        raw110_status='N/A: no adequate continuous eligible real paint reference in frozen source-only candidate set',
        bench_mother_domains_or_metrics_modified=False,native_calls=0,GPU_operations=0,optimizer_updates=0,power_operations=0,human_verdict=None)
    write(OUT/'results.json',result);print(json.dumps(dict(output=str(OUT),seconds=result['seconds'],rows=len(rows),source_ROIs=roi_info)),flush=True)

if __name__=='__main__':main()
