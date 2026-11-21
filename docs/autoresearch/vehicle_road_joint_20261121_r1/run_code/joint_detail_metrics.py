"""复用原道路细节指标，在固定可见域评价车辆与道路，附 masked SSIM。"""
import numpy as np
from scipy import ndimage as ndi
from road_detail_metrics import DetailConfig, ResolutionSpec, TileLayout, evaluate_detail, _strict_core

CONFIG=DetailConfig(pixel_scales=(1.,2.,4.),flat_hf_thresholds=(2/255,),
                    edge_thresholds=(2/255,),edge_tolerances_pixels=(1.,),
                    edge_angle_tolerances_degrees=(30.,),worst_tiles=0)

def measure(pred, target, mask, *, teacher_role='real_observation', context=None):
    """HWC [0,1] sRGB。mask 是固定评分域，不依预测质量筛选像素。"""
    pred=np.asarray(pred);target=np.asarray(target);mask=np.asarray(mask)
    assert pred.ndim==3 and pred.shape==target.shape and pred.shape[-1]==3
    h,w=pred.shape[:2]
    assert pred.dtype.kind=='f' and target.dtype.kind=='f'
    assert mask.dtype==bool and mask.shape==(h,w)
    if context is None:context=mask
    assert np.isfinite(pred[context]).all() and np.isfinite(target[context]).all()
    delta=pred[mask].astype(float)-target[mask].astype(float)
    report=evaluate_detail(pred,target,context_mask=context,score_mask=mask,
        resolution=ResolutionSpec(w,h,'native1152','srgb'),tile_layout=TileLayout(h,w),
        teacher_role=teacher_role,config=CONFIG,input_conditioning={'domain':'fixed_source_mask','alpha_filter':False})
    # 与 HF 相同，只在完整卷积核处评估局部结构，不把 mask 外背景带进统计。
    core=mask&_strict_core(context,5)
    mu=[];var=[];cross=[]
    for a in [pred,target]:
        m=ndi.gaussian_filter(a.astype(float),(1.5,1.5,0),mode='reflect',truncate=10/3)
        mu.append(m)
        var.append(np.maximum(0,ndi.gaussian_filter(a.astype(float)**2,(1.5,1.5,0),mode='reflect',truncate=10/3)-m*m))
    cross=ndi.gaussian_filter(pred.astype(float)*target.astype(float),(1.5,1.5,0),mode='reflect',truncate=10/3)-mu[0]*mu[1]
    ssim=((2*mu[0]*mu[1]+.01**2)*(2*cross+.03**2))/((mu[0]**2+mu[1]**2+.01**2)*(var[0]+var[1]+.03**2))
    curve=report['edges']['threshold_curves'][0]['curves'][0]
    p,r=curve['precision'],curve['recall']
    row=dict(pixels=int(mask.sum()),RGB_MAE=float(abs(delta).mean()) if delta.size else None,
             RGB_RMSE=float(np.sqrt((delta*delta).mean())) if delta.size else None,
             masked_SSIM=float(ssim[core].mean()) if core.any() else None,SSIM_core_pixels=int(core.sum()),
             edge_F1=None if p is None or r is None else (2*p*r/(p+r) if p+r else 0),
             edge_reference='appearance_edges_not_lane_geometry_truth')
    for scale in report['scales']:
        suffix=str(int(scale['sigma_pixels']))
        for key in ['HF_cosine','HF_energy_ratio','HF_MAE','HF_RMSE']:
            row[key+'_'+suffix]=scale['core'][key]
        row['HF_core_pixels_'+suffix]=scale['core_pixels']
    return row,report
