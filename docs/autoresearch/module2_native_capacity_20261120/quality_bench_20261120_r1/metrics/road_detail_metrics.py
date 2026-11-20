"""域安全的 CPU 细节对应指标；无总分、通过阈值或自动样本剔除。"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ResolutionSpec:
    width: int
    height: int
    pixel_space: str
    signal_space: str = 'srgb'


@dataclass(frozen=True)
class TileLayout:
    height: int
    width: int
    origin_y: int = 0
    origin_x: int = 0


@dataclass(frozen=True)
class DetailConfig:
    pixel_scales: tuple = (1., 2., 4.)
    gaussian_truncate: float = 3.
    flat_hf_thresholds: tuple = (1/255, 2/255, 4/255)
    edge_thresholds: tuple = (1/255, 2/255, 4/255)
    edge_tolerances_pixels: tuple = (0., 1., 2.)
    edge_angle_tolerances_degrees: tuple = (15., 30., 45.)
    worst_tiles: int = 5


def _mask(value, shape, name):
    a = np.asarray(value)
    if a.dtype != bool or a.shape != shape:
        raise ValueError(name+' must be a bool HW array')
    return a


def _image(value, shape):
    a = np.asarray(value)
    if a.shape != (*shape, 3):
        raise ValueError('RGB input must be HWC; no implicit CHW/resize')
    if a.dtype == np.uint8:
        return a.astype(np.float64)/255
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError('RGB must be floating point or uint8; no value-based normalization')
    return a.astype(np.float64, copy=False)


def _summary(values):
    a = np.asarray(values, np.float64)
    if not a.size:
        return dict(count=0, mean=None, p05=None, p10=None, p50=None, p90=None, p95=None, max=None)
    return dict(count=int(a.size), mean=float(a.mean()), p05=float(np.quantile(a,.05)),p10=float(np.quantile(a,.1)),p50=float(np.quantile(a,.5)),
                p90=float(np.quantile(a,.9)), p95=float(np.quantile(a,.95)), max=float(a.max()))


def _strict_core(context, radius):
    return ndi.minimum_filter(context.astype(np.uint8), size=2*radius+1,
                              mode='constant', cval=0).astype(bool)


def _masked_hf(image, context, sigma, truncate):
    mass = ndi.gaussian_filter(context.astype(np.float64), sigma, mode='constant', cval=0, truncate=truncate)
    clean = np.where(context[..., None], image, 0.)
    numerator = np.stack([ndi.gaussian_filter(clean[..., c], sigma, mode='constant',
                         cval=0, truncate=truncate) for c in range(3)], -1)
    low = np.divide(numerator, mass[..., None], out=np.zeros_like(numerator), where=mass[..., None]>0)
    return np.where(context[..., None], clean-low, 0.), mass


def _hf_stats(pred, teacher, mask):
    n = int(mask.sum())
    if not n:
        return dict(status='no_scored_core', pixels=0, HF_MAE=None, HF_RMSE=None,
                    HF_cosine=None, reference_HF_RMS=None, predicted_HF_RMS=None,
                    HF_energy_ratio=None, HF_relative_RMSE=None, excess_HF_energy=None,
                    missing_HF_energy=None, reference_energy_sum=0.)
    p, t = pred[mask], teacher[mask]
    ep, et = float(np.mean(p*p)), float(np.mean(t*t))
    error = float(np.mean((p-t)**2))
    # 只用于浮点零判定，不是画质接受门。
    zero = np.finfo(np.float64).eps**2 * 64
    status = 'defined'
    if et <= zero:
        status = 'reference_zero_energy'
    elif ep <= zero:
        status = 'prediction_zero_reference_nonzero'
    cosine = float(np.clip(np.mean(p*t)/math.sqrt(ep*et),-1,1)) if ep>zero and et>zero else None
    return dict(status=status, pixels=n, HF_MAE=float(np.mean(abs(p-t))), HF_RMSE=math.sqrt(error),
        HF_cosine=cosine, reference_HF_RMS=math.sqrt(et), predicted_HF_RMS=math.sqrt(ep),
        HF_energy_ratio=ep/et if et>zero else None,
        HF_relative_RMSE=math.sqrt(error/et) if et>zero else None,
        excess_HF_energy=max(0.,ep-et), missing_HF_energy=max(0.,et-ep),
        reference_energy_sum=et*n*3 if et>zero else 0.)


def _flat_stats(pred_hf, teacher_hf, mask, thresholds):
    magnitude = np.sqrt(np.mean(teacher_hf**2, axis=-1))
    return [dict(teacher_HF_RMS_cut=float(cut), pixels=int((mask&(magnitude<=cut)).sum()),
                 predicted_HF_RMS=float(np.sqrt(np.mean(pred_hf[mask&(magnitude<=cut)]**2)))
                    if np.any(mask&(magnitude<=cut)) else None,
                 teacher_HF_RMS=float(np.sqrt(np.mean(teacher_hf[mask&(magnitude<=cut)]**2)))
                    if np.any(mask&(magnitude<=cut)) else None)
            for cut in thresholds]


def _tiles(shape, layout):
    if layout.height <= 0 or layout.width <= 0 or (layout.origin_y,layout.origin_x)!=(0,0):
        raise ValueError('tile layout must have positive size and fixed full-image origin(0,0)')
    for y in range(0,shape[0],layout.height):
        for x in range(0,shape[1],layout.width):
            yield (y//layout.height,x//layout.width), (y,min(y+layout.height,shape[0]),x,min(x+layout.width,shape[1]))


def _tile_aggregate(rows, cuts, worst_count):
    valid = [r for r in rows if r['core']['pixels']]
    keys = ['HF_MAE','HF_RMSE','HF_cosine','HF_relative_RMSE','excess_HF_energy','predicted_HF_RMS']
    macro = {key: _summary([r['core'][key] for r in valid if r['core'][key] is not None]) for key in keys}
    total_reference = sum(r['core']['reference_energy_sum'] for r in valid)
    weighted = sum(r['core']['HF_MAE']*r['core']['reference_energy_sum'] for r in valid)/total_reference if total_reference>0 else None
    strata = []
    for cut in cuts:
        detail = [r for r in valid if r['core']['reference_HF_RMS']>=cut]
        worst = sorted(detail, key=lambda r:r['core']['HF_RMSE'], reverse=True)[:worst_count]
        strata.append(dict(reference_HF_RMS_min=float(cut), eligible_tiles=len(detail),
            macro={key:_summary([r['core'][key] for r in detail if r['core'][key] is not None]) for key in keys},
            worst_tiles=[dict(tile_id=r['tile_id'], bounds_y0_y1_x0_x1=r['bounds_y0_y1_x0_x1'],
                             core_pixels=r['core']['pixels'], metrics=r['core']) for r in worst]))
    worst = sorted(valid,key=lambda r:r['core']['HF_RMSE'],reverse=True)[:worst_count]
    return dict(total_fixed_tiles=len(rows), nonempty_core_tiles=len(valid),
        empty_score_tiles=sum(r['score_pixels']==0 for r in rows),
        scored_but_no_core_tiles=sum(r['score_pixels']>0 and r['core']['pixels']==0 for r in rows),
        cosine_undefined_tiles=sum(r['core']['HF_cosine'] is None for r in valid),
        macro=macro, reference_energy_weighted_tile_HF_MAE=weighted,
        reference_detail_strata=strata,
        worst_tiles=[dict(tile_id=r['tile_id'],bounds_y0_y1_x0_x1=r['bounds_y0_y1_x0_x1'],metrics=r['core'])for r in worst])


def _gradient_edges(image, context):
    clean = np.where(context[...,None],image,0.)
    gx = np.stack([ndi.sobel(clean[...,c],axis=1,mode='constant')/8 for c in range(3)],-1)
    gy = np.stack([ndi.sobel(clean[...,c],axis=0,mode='constant')/8 for c in range(3)],-1)
    a, b, d = np.mean(gx*gx,-1), np.mean(gx*gy,-1), np.mean(gy*gy,-1)
    strength = np.sqrt(np.maximum(0.,(a+d+np.sqrt((a-d)**2+4*b*b))/2))
    angle = .5*np.arctan2(2*b,a-d)  # 无向边缘法线，mod pi。
    yy,xx = np.indices(context.shape)
    forward = ndi.map_coordinates(strength,[yy+np.sin(angle),xx+np.cos(angle)],order=1,mode='constant')
    backward = ndi.map_coordinates(strength,[yy-np.sin(angle),xx-np.cos(angle)],order=1,mode='constant')
    nms = (strength>=forward)&(strength>backward)
    return strength, angle, nms, _strict_core(context,2)


def _nearest_angles(edges, angle, query, query_angle):
    if not query.any():
        return np.empty(0),np.empty(0)
    if not edges.any():
        return np.full(int(query.sum()),np.inf),np.full(int(query.sum()),np.nan)
    distance, indices = ndi.distance_transform_edt(~edges, return_indices=True)
    near = angle[indices[0][query],indices[1][query]]
    error = np.arccos(np.clip(np.abs(np.cos(query_angle[query]-near)),0,1))*180/np.pi
    return distance[query],error


def _edge_stats(pred, teacher, pred_angle, teacher_angle, tolerances, angle_tolerances, matches=None):
    npred, nref = int(pred.sum()),int(teacher.sum())
    if matches is None:
        pd,pa = _nearest_angles(teacher,teacher_angle,pred,pred_angle)
        td,ta = _nearest_angles(pred,pred_angle,teacher,teacher_angle)
    else:
        pd,pa,td,ta=matches
        if len(pd)!=npred or len(td)!=nref:
            raise ValueError('global match vectors do not match local edge identity')
    curves=[]
    for radius in tolerances:
        precision=float((pd<=radius).sum()/npred) if npred else None
        recall=float((td<=radius).sum()/nref) if nref else None
        # 若另一边为空，距离向量为空，分母仍保留：precision/recall=0，不填满分。
        orientation=[]
        known_p,known_t=np.isfinite(pa),np.isfinite(ta)
        for degrees in angle_tolerances:
            orientation.append(dict(max_normal_error_degrees=float(degrees),
                precision=float(((pd<=radius)&(pa<=degrees)).sum()/known_p.sum()) if known_p.any() else None,
                recall=float(((td<=radius)&(ta<=degrees)).sum()/known_t.sum()) if known_t.any() else None,
                predicted_orientation_comparable_edges=int(known_p.sum()),reference_orientation_comparable_edges=int(known_t.sum())))
        curves.append(dict(tolerance_pixels=float(radius),precision=precision,recall=recall,
            matched_predicted=int((pd<=radius).sum()),matched_reference=int((td<=radius).sum()),
            predicted_normal_error_degrees=_summary(pa[(pd<=radius)&known_p]),
            reference_normal_error_degrees=_summary(ta[(td<=radius)&known_t]),orientation_curves=orientation))
    return dict(reference_edges=nref,predicted_edges=npred,
        status='defined' if nref or npred else 'no_local_edges',
        reference_to_prediction_distance_pixels=_summary(td[np.isfinite(td)]),prediction_to_reference_distance_pixels=_summary(pd[np.isfinite(pd)]),
        reference_distance_unavailable_edges=int((~np.isfinite(td)).sum()),
        prediction_distance_unavailable_edges=int((~np.isfinite(pd)).sum()),
        curves=curves)


def _edge_tile_aggregate(rows, curve_count, worst_count):
    result=[]
    for index in range(curve_count):
        precision=[r['metrics']['curves'][index]['precision']for r in rows if r['metrics']['curves'][index]['precision']is not None]
        recall=[r['metrics']['curves'][index]['recall']for r in rows if r['metrics']['curves'][index]['recall']is not None]
        worst=sorted([r for r in rows if r['metrics']['reference_edges']],key=lambda r:r['metrics']['curves'][index]['recall'])[:worst_count]
        worst_precision=sorted([r for r in rows if r['metrics']['predicted_edges']],key=lambda r:r['metrics']['curves'][index]['precision'])[:worst_count]
        result.append(dict(tolerance_index=index,precision_macro=_summary(precision),recall_macro=_summary(recall),
            total_tiles=len(rows),empty_score_tiles=sum(r['score_pixels']==0 for r in rows),
            no_reference_edge_tiles=sum(r['metrics']['reference_edges']==0 for r in rows),
            no_matched_prediction_on_reference_tiles=sum(r['metrics']['reference_edges']>0 and r['metrics']['curves'][index]['recall']==0 for r in rows),
            zero_local_predicted_edge_tiles=sum(r['metrics']['predicted_edges']==0 for r in rows),
            reference_to_prediction_mean_distance_macro=_summary([r['metrics']['reference_to_prediction_distance_pixels']['mean']for r in rows if r['metrics']['reference_to_prediction_distance_pixels']['mean']is not None]),
            prediction_to_reference_mean_distance_macro=_summary([r['metrics']['prediction_to_reference_distance_pixels']['mean']for r in rows if r['metrics']['prediction_to_reference_distance_pixels']['mean']is not None]),
            worst_recall_tiles=[dict(tile_id=r['tile_id'],bounds_y0_y1_x0_x1=r['bounds_y0_y1_x0_x1'],
                reference_edges=r['metrics']['reference_edges'],predicted_edges=r['metrics']['predicted_edges'],
                metrics=r['metrics']['curves'][index])for r in worst],
            worst_precision_tiles=[dict(tile_id=r['tile_id'],bounds_y0_y1_x0_x1=r['bounds_y0_y1_x0_x1'],
                reference_edges=r['metrics']['reference_edges'],predicted_edges=r['metrics']['predicted_edges'],
                metrics=r['metrics']['curves'][index])for r in worst_precision]))
    return result


def evaluate_detail(pred_rgb, teacher_rgb, *, context_mask, score_mask, resolution,
                    tile_layout, teacher_role, config=DetailConfig(), edge_reference=None,
                    color_space=None, input_conditioning=None):
    """同分辨率 HWC RGB；不做resize/自动曝光、不修改mask、不输出PASS/总分。

    teacher_role: real_observation | frozen_pseudo_teacher | synthetic_known。
    edge_reference可给mask/normal_angle_radians/role；缺少可验证结构时只报teacher外观边缘。
    """
    if not isinstance(resolution,ResolutionSpec) or not resolution.pixel_space or resolution.signal_space not in ('srgb','premult_srgb','linear_rgb'):
        raise ValueError('explicit resolution/pixel_space/signal_space is required')
    if teacher_role not in ('real_observation','frozen_pseudo_teacher','synthetic_known'):
        raise ValueError('teacher provenance role required; pseudo teacher is not geometry GT')
    shape=(resolution.height,resolution.width)
    context=_mask(context_mask,shape,'context_mask');score=_mask(score_mask,shape,'score_mask')
    pred,teacher=_image(pred_rgb,shape),_image(teacher_rgb,shape)
    if not np.isfinite(pred[context]).all() or not np.isfinite(teacher[context]).all():
        raise ValueError('nonfinite RGB inside context must not be silently removed')
    if any(s<=0 for s in config.pixel_scales) or config.gaussian_truncate<=0 or any(t<=0 for t in config.edge_thresholds):
        raise ValueError('positive measurement scales/thresholds required')
    if any(t<0 for t in (*config.flat_hf_thresholds,*config.edge_tolerances_pixels)):
        raise ValueError('measurement cuts/tolerances must be nonnegative')
    tile_boxes=list(_tiles(shape,tile_layout));scored=score&context
    color_space=resolution.signal_space if color_space is None else color_space
    if color_space not in ('srgb','premult_srgb','linear_rgb','stable_alpha_unpremult_srgb'):
        raise ValueError('explicit supported color_space required')
    result=dict(schema='road_detail_correspondence_metrics.v1',resolution=asdict(resolution),color_space=color_space,
        input_conditioning={} if input_conditioning is None else dict(input_conditioning),
        teacher_role=teacher_role,config=asdict(config),tile_layout=asdict(tile_layout),
        units='all lengths/scales/tolerances in the declared input pixel grid; no automatic resize',
        acceptance_policy='descriptive vector only; thresholds define measurement curves, not passing gates',
        fixed_score_pixels=int(score.sum()),score_outside_context_pixels=int((score&~context).sum()),
        context_scored_pixels=int(scored.sum()),context_pixels=int(context.sum()),scales=[])
    for sigma in config.pixel_scales:
        hp,mass=_masked_hf(pred,context,sigma,config.gaussian_truncate)
        ht,_=_masked_hf(teacher,context,sigma,config.gaussian_truncate)
        radius=int(config.gaussian_truncate*sigma+.5);core=scored&_strict_core(context,radius)
        boundary=scored&~core
        rows=[]
        for tile_id,(y0,y1,x0,x1) in tile_boxes:
            sl=np.s_[y0:y1,x0:x1];m=core[sl]
            rows.append(dict(tile_id=list(tile_id),bounds_y0_y1_x0_x1=[y0,y1,x0,x1],
                score_pixels=int(score[sl].sum()),core=_hf_stats(hp[sl],ht[sl],m),
                boundary_pixels=int(boundary[sl].sum()),
                flat_spurious=_flat_stats(hp[sl],ht[sl],m,config.flat_hf_thresholds)))
        result['scales'].append(dict(sigma_pixels=float(sigma),finite_kernel_radius=radius,
            full_score_pixels=int(score.sum()),core_pixels=int(core.sum()),boundary_pixels=int(boundary.sum()),
            core=_hf_stats(hp,ht,core),boundary_partial_context=_hf_stats(hp,ht,boundary),
            boundary_context_mass=_summary(mass[boundary]),
            teacher_flat_spurious=_flat_stats(hp,ht,core,config.flat_hf_thresholds),
            tile_aggregate=_tile_aggregate(rows,config.flat_hf_thresholds,config.worst_tiles),tiles=rows))
    ps,pa,pn,edgecore=_gradient_edges(pred,context)
    ts,ta,tn,_=_gradient_edges(teacher,context)
    edgecore &= scored
    reference_role='teacher_appearance_edges'
    explicit_reference=None
    if edge_reference is not None:
        reference_role=edge_reference.get('role')
        if reference_role not in ('human_verified_structure','teacher_verified_structure','synthetic_known_structure'):
            raise ValueError('explicit structure edges need a verifiable source role; generated lanes are not geometry GT')
        explicit_reference=_mask(edge_reference['mask'],shape,'edge_reference.mask')
        if 'normal_angle_radians' in edge_reference:
            ta=np.asarray(edge_reference['normal_angle_radians'],np.float64)
            if ta.shape!=shape or not np.isfinite(ta[explicit_reference]).all():
                raise ValueError('reference normal angles invalid')
        else:
            ta=np.where(ts>np.finfo(np.float64).eps,ta,np.nan)
    edge_rows=[]
    for threshold in config.edge_thresholds:
        reference=(tn&(ts>=threshold) if explicit_reference is None else explicit_reference)&edgecore
        predicted=pn&(ps>=threshold)&edgecore
        pd,pangle=_nearest_angles(reference,ta,predicted,pa)
        td,tangle=_nearest_angles(predicted,pa,reference,ta)
        stats=_edge_stats(predicted,reference,pa,ta,config.edge_tolerances_pixels,config.edge_angle_tolerances_degrees,
                          matches=(pd,pangle,td,tangle))
        pred_y,pred_x=np.nonzero(predicted);ref_y,ref_x=np.nonzero(reference)
        tiles=[]
        for tile_id,(y0,y1,x0,x1) in tile_boxes:
            sl=np.s_[y0:y1,x0:x1]
            pselect=(pred_y>=y0)&(pred_y<y1)&(pred_x>=x0)&(pred_x<x1)
            tselect=(ref_y>=y0)&(ref_y<y1)&(ref_x>=x0)&(ref_x<x1)
            tiles.append(dict(tile_id=list(tile_id),bounds_y0_y1_x0_x1=[y0,y1,x0,x1],
                score_pixels=int(score[sl].sum()),core_pixels=int(edgecore[sl].sum()),
                metrics=_edge_stats(predicted[sl],reference[sl],pa[sl],ta[sl],
                    config.edge_tolerances_pixels,config.edge_angle_tolerances_degrees,
                    matches=(pd[pselect],pangle[pselect],td[tselect],tangle[tselect]))))
        edge_rows.append(dict(strength_threshold=float(threshold),**stats,tiles=tiles,
            tile_aggregate=_edge_tile_aggregate(tiles,len(config.edge_tolerances_pixels),config.worst_tiles)))
    result['edges']=dict(reference_role=reference_role,
        semantic_lane_geometry_truth_claim=False,
        detector='RGB structure tensor dominant gradient + interpolated nonmaximum suppression; unsigned normals modulo pi',
        matching='once over the full fixed scored domain, then aggregate each reference/prediction by its own tile; cross-tile matches remain valid',
        context_core_pixels=int(edgecore.sum()),boundary_pixels=int(scored.sum()-edgecore.sum()),
        threshold_curves=edge_rows)
    return result


def assert_comparable(reports):
    """禁止把1920原像素和480像素结果静默合池。"""
    if not reports:
        raise ValueError('no reports')
    first=reports[0]
    for report in reports[1:]:
        for key in ('resolution','color_space','input_conditioning','config','tile_layout'):
            if report[key]!=first[key]:
                raise ValueError('incompatible measurement basis: '+key)


def rank_cases(cases, *, metric_path, higher_is_worse):
    """显式指标排序并保留全部N/A病例；不生成总分或剔除失败。"""
    pairs=list(cases.items());assert_comparable([r for _,r in pairs])
    rows=[]
    for name,report in pairs:
        value=report
        for key in metric_path:value=value[key]
        if value is not None and not isinstance(value,(float,int)):
            raise ValueError('ranking requires an explicit scalar metric path')
        rows.append(dict(case_id=name,value=value,status='N/A' if value is None else'defined'))
    defined=sorted([r for r in rows if r['value']is not None],key=lambda r:r['value'],reverse=higher_is_worse)
    return dict(metric_path=list(metric_path),higher_is_worse=bool(higher_is_worse),
                cases=defined+[r for r in rows if r['value']is None],excluded_cases=0)
