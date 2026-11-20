"""CPU synthetic-paint endpoint metrics. No renderer, training or unknown GT."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def summary(a):
    a=np.asarray(a,np.float64)
    return dict(n=int(a.size),mean=float(a.mean()) if a.size else None,p95=float(np.quantile(a,.95)) if a.size else None,max=float(a.max()) if a.size else None)

def metric_rgb(a,b,mask):
    n=int(mask.sum())
    if not n:return dict(pixels=0,MAE=None,RMSE=None,status='N/A_no_pixels')
    d=np.asarray(a,np.float64)[:,mask]-np.asarray(b,np.float64)[:,mask]
    return dict(pixels=n,MAE=float(np.abs(d).mean()),RMSE=float(np.sqrt((d*d).mean())),status='defined')

def paint_metrics(P,*,readonly,weight,asphalt,paint_rgb,paint_fraction,qualified_mask,dash_ids=None):
    """Known analytic paint only; preserve old clipped-fraction contrast0.90.

    Width is descriptive horizontal native-pixel half-height width against the
    same AA target, never mislabeled the physical outer width of soft-edged paint.
    """
    P=np.asarray(P,np.float64);readonly=np.asarray(readonly,np.float64);weight=np.asarray(weight,np.float64)
    asphalt=np.asarray(asphalt,np.float64);paint_fraction=np.asarray(paint_fraction,np.float64)
    mask=np.asarray(qualified_mask,bool)
    if P.shape!=readonly.shape or P.shape!=asphalt.shape or P.shape[0]!=3 or P.shape[1:]!=mask.shape or weight.shape!=mask.shape:raise ValueError('paint CHW/HW shape mismatch')
    if np.any(mask&(weight<.95)):raise ValueError('paint capability layer must use predeclared We>=.95')
    d=np.asarray(paint_rgb,np.float64)[:,None,None]-asphalt
    color=np.divide(P-readonly,weight[None],out=np.zeros_like(P),where=weight[None]>1e-12)
    denom=np.sum(d*d,axis=0)
    raw=np.divide(np.sum((color-asphalt)*d,axis=0),denom,out=np.zeros_like(weight),where=denom>1e-12)
    fraction=np.clip(raw,0.,1.)
    binary=fraction>=.5;known=paint_fraction>=.5
    edge_valid=binary_erosion(mask,iterations=1)
    ref=(known&~binary_erosion(known))&edge_valid
    pred=(binary&~binary_erosion(binary))&edge_valid
    nref,npred=int(ref.sum()),int(pred.sum())
    if nref:
        precision=float((distance_transform_edt(~ref)[pred]<=1.).mean()) if npred else None
        recall=float((distance_transform_edt(~pred)[ref]<=1.).mean()) if npred else 0.
        F1=(2*precision*recall/(precision+recall) if precision is not None and precision+recall else 0.)
        ref_dist=summary(distance_transform_edt(~pred)[ref]) if npred else summary([])
        pred_dist=summary(distance_transform_edt(~ref)[pred]) if npred else summary([])
    else:precision=recall=F1=None;ref_dist=pred_dist=summary([])
    core=mask&(paint_fraction>=.9);flank=mask&(paint_fraction<=.1)
    truth=float(paint_fraction[core].mean()-paint_fraction[flank].mean()) if core.any() and flank.any() else None
    contrast=float((fraction[core].mean()-fraction[flank].mean())/truth) if truth is not None and truth>0 else None
    yy,xx=np.indices(mask.shape);errors=[];missing=0;groups=0
    if dash_ids is not None:
        dash_ids=np.asarray(dash_ids)
        for key in np.unique(dash_ids[mask&known]):
            region=mask&(dash_ids==key);t=paint_fraction[region];p=fraction[region]
            if not t.size or t.sum()<=0:continue
            groups+=1
            if p.sum()<=1e-12:missing+=1;continue
            xy=np.stack([xx[region],yy[region]],-1)
            errors.append(float(np.linalg.norm((xy*t[:,None]).sum(0)/t.sum()-(xy*p[:,None]).sum(0)/p.sum())))
    widths=[];missing_sections=0
    for y in np.flatnonzero((mask&known).any(1)):
        refx=np.flatnonzero(mask[y]&known[y])
        if len(refx)<3:continue
        lo,hi=int(refx.min()),int(refx.max())
        # A complete source-only cross section needs asphalt on both sides.
        if not (np.any(mask[y,:lo]&(paint_fraction[y,:lo]<=.1)) and np.any(mask[y,hi+1:]&(paint_fraction[y,hi+1:]<=.1))):continue
        px=np.flatnonzero(mask[y]&binary[y]);rw=hi-lo+1;pw=int(px.max()-px.min()+1) if len(px) else 0
        missing_sections+=int(not len(px));widths.append((rw,pw))
    width_error=np.array([abs(p-r) for r,p in widths],float)
    evaluable=bool(mask.any() and core.any() and flank.any() and nref and groups)
    old_combined=(contrast is not None and contrast>=.90 and F1 is not None and F1>=.9 and bool(errors) and max(errors)<=1. and missing==0) if evaluable else None
    return dict(schema='known_world_paint_endpoint.v1',role='synthetic_known_not_hidden_road_GT',qualified_pixels=int(mask.sum()),
        qualification='source-legal exact canonical surface, ideal available, common alpha>=.95, eligible We>=.95 and historical>=3px/Nyquist qualification',
        paint_core=int(core.sum()),asphalt_flank=int(flank.sum()),reference_edges=nref,predicted_edges=npred,
        edge_precision_1px=precision,edge_recall_1px=recall,edge_F1_1px=F1,
        reference_to_prediction_distance_px=ref_dist,prediction_to_reference_distance_px=pred_dist,
        contrast_retention=contrast,old_contrast_threshold=.90,contrast_gate=(contrast>=.9) if contrast is not None else None,
        known_dash_fragments=groups,missing_predicted_fragments=missing,line_center_error_max_px=max(errors) if errors else None,
        halfheight_width=dict(role='descriptive matched native-horizontal cross sections; not physical outer paint width or a new pass criterion',
            sections=len(widths),missing_predicted_sections=missing_sections,absolute_error_px=summary(width_error),
            reference_mean_px=float(np.mean([r for r,p in widths])) if widths else None,predicted_mean_px=float(np.mean([p for r,p in widths])) if widths else None),
        raw_fraction_outside_0_1=int(((raw<0)|(raw>1))[mask].sum()),raw_fraction_range=[float(raw[mask].min()),float(raw[mask].max())] if mask.any() else None,
        clipped_fraction='clip[0,1] is retained from the historical ideal detail evaluator; raw overshoot is also reported',
        status='defined' if evaluable else 'N/A_insufficient_qualified_reference',old_combined_detail_gate=old_combined,
        negative_capacity_certificate=False,production_promotion=False)
