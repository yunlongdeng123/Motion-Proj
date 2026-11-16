"""Scene-independent native masks -> geometry/material camera inputs.

No detector, identity matcher, bbox rasterizer, crop search, or model call lives
here. Pixel proposals and an explicit target enter once; every training/query
view uses the same rendering path. RGB outside valid material support is erased
before resampling. Accepted masks are never gated by same-frame LiDAR returns.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, label


def _mask(value, shape, name):
    if value is None:
        return np.zeros(shape, bool)
    raw=np.asarray(value)
    if np.issubdtype(raw.dtype,np.floating) and (not np.isfinite(raw).all() or not np.isin(raw,[0.,1.]).all()):
        raise ValueError(f'{name} is a binary mask; pass soft confidence through ShadowField, not effect')
    result=np.asarray(value,bool)
    if result.shape!=shape:
        raise ValueError(f'{name} must use the native camera grid')
    return result.copy()


def _band(mask, radius):
    if radius<=0:return np.zeros_like(mask)
    return binary_dilation(mask,iterations=radius)&~binary_erosion(mask,iterations=1)


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    return value


def prepare_native_layers(rgb, road_mask, target_core, *, semantic_foreground=None,
                          accepted_foreground=None, candidate_foreground=None,
                          foreground_uncertainty=None, target_uncertainty=None,
                          effect=None, retained_core=None, K=None,
                          sam_confidence=None, effect_confidence=None, depth_hint=None,
                          pseudo_loss_mask=None, supervision_rgb=None, supervision_valid_mask=None, provenance=None,
                          uncertainty_radius=2, retained_guard=4, neutral_rgb=(127,127,127),
                          independent_regions=None):
    """Prepare native roles without selecting a view or consuming hidden-road GT.

    candidate_foreground is a possible-foreground pixel union, not certification.
    retained_core/K must describe nonedited objects, not unresolved foreground.
    pseudo_loss_mask is permitted only on independently visible source road.
    Its source RGB is kept only in the separate sanitized supervision image.
    """
    start=time.perf_counter()
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('rgb must be native uint8 sRGB HxWx3')
    shape=rgb.shape[:2]
    road=_mask(road_mask,shape,'road_mask')
    target=_mask(target_core,shape,'target_core')
    if not target.any():raise ValueError('an explicit nonempty target pixel mask is required')
    semantic=_mask(semantic_foreground,shape,'semantic_foreground')
    accepted=_mask(accepted_foreground,shape,'accepted_foreground')
    candidate=_mask(candidate_foreground,shape,'candidate_foreground')
    core=semantic|accepted
    possible=core|candidate
    fg_uncertainty=(_mask(foreground_uncertainty,shape,'foreground_uncertainty')
                    |(candidate&~core)|_band(possible,int(uncertainty_radius)))
    target_uncertainty=(_mask(target_uncertainty,shape,'target_uncertainty')
                        |(binary_dilation(target,iterations=int(uncertainty_radius))&~target if uncertainty_radius>0 else False))
    effect=_mask(effect,shape,'effect')
    retained=_mask(retained_core,shape,'retained_core')
    protected=(_mask(K,shape,'K') if K is not None else
               binary_dilation(retained,iterations=int(retained_guard)) if retained_guard>0 else retained.copy())
    ownership_conflict=target&retained
    protected &= ~target
    unknown_core=(target|target_uncertainty)&~protected
    unknown=(unknown_core|effect)&~protected
    uncertainty=fg_uncertainty|target_uncertainty|ownership_conflict
    observed_road=road&~possible&~fg_uncertainty&~unknown_core
    cmat=observed_road&~unknown
    inferred_road=unknown_core.copy()
    support=observed_road|inferred_road
    masks=dict(target_core=target,target_uncertainty=target_uncertainty,target_unknown=unknown,
               rgb_unknown_core=unknown_core,material_contaminated_effect=effect,
               physical_unknown=unknown_core,
               geometry_observed_valid=~unknown_core,geometry_road_visible=observed_road,
               road_geometry_observed=observed_road,
               effect_inferred=effect,retained_core=retained,K=protected,
               ownership_conflict=ownership_conflict,foreground_core=core,
               foreground_possible=possible,foreground_uncertainty=fg_uncertainty,
               uncertainty=uncertainty,road_observed=observed_road,
               road_inferred=inferred_road,road_support=support,C_mat=cmat,
               visible_support=cmat,retained_context=~unknown)
    confidence={}
    for name,array in [('sam_confidence',sam_confidence),('effect_confidence',effect_confidence)]:
        if array is not None:
            value=np.asarray(array,np.float32)
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError(f'{name} must be a finite native confidence grid')
            confidence[name]=np.clip(value,0,1)
    loss_requested=_mask(pseudo_loss_mask,shape,'pseudo_loss_mask')
    supervision_valid=(_mask(supervision_valid_mask,shape,'supervision_valid_mask') if supervision_valid_mask is not None
                       else road&~possible&~fg_uncertainty)
    loss=loss_requested&supervision_valid
    target_sanitized=None
    if pseudo_loss_mask is not None:
        clean=rgb if supervision_rgb is None else np.asarray(supervision_rgb)
        if clean.shape!=rgb.shape or clean.dtype!=np.uint8:
            raise ValueError('supervision_rgb must be separate native uint8 RGB')
        target_sanitized=clean.copy()
        target_sanitized[possible|fg_uncertainty]=neutral_rgb
        target_sanitized[loss]=clean[loss]
    depth=None
    if depth_hint is not None:
        value=np.asarray(depth_hint,np.float32)
        if value.shape!=shape:raise ValueError('depth_hint must use native camera grid')
        valid=np.isfinite(value)&(value>0)&~unknown_core
        depth=np.where(valid,value,0)
        masks['depth_valid']=valid
    receipt=dict(schema='native_input_roles.v1',native_hw=list(shape),
                 target_pixels=int(target.sum()),unknown_pixels=int(unknown.sum()),
                 foreground_core_pixels=int(core.sum()),foreground_possible_pixels=int(possible.sum()),
                 material_valid_pixels=int(cmat.sum()),uncertainty_pixels=int(uncertainty.sum()),
                 target_retained_identity_conflict_pixels=int(ownership_conflict.sum()),
                 accepted_masks_require_current_lidar=False,bbox_rasterized_into_diffusion_mask=False,
                 material_mask_requires_actor_identity=False,
                 road_under_target_is_inferred_support_not_GT=True,
                 geometry_policy='retain all nontarget RGB, erase target/effect only',
                 material_policy='reliable observed road excluding all possible foreground, uncertainty and target',
                 source_provenance=_plain(provenance or {}),prepare_seconds=time.perf_counter()-start)
    result=dict(rgb=rgb.copy(),masks=masks,confidence=confidence,depth_hint=depth,
                supervision_rgb=None if target_sanitized is None else clean.copy(),
                target_sanitized=target_sanitized,loss_mask=loss,neutral_rgb=np.asarray(neutral_rgb,np.uint8),receipt=receipt)
    if independent_regions is not None:
        from .independent_regions import apply_independent_regions
        result=apply_independent_regions(result,**independent_regions)
    return result


def _plan(plan, native_hw):
    if not isinstance(plan,dict):plan=vars(plan)
    wh=plan.get('output_wh')
    if wh is None:raise ValueError('view plan needs output_wh')
    width,height=map(int,wh)
    if min(width,height)<=0:raise ValueError('view resolution must be positive')
    matrix=plan.get('native_from_input')
    window=plan.get('window_xyxy',plan.get('native_crop_xyxy'))
    if matrix is None:
        if window is None:raise ValueError('view plan needs native_from_input or a native window')
        x0,y0,x1,y1=map(float,window)
        matrix=[[float(x1-x0)/width,0,x0],[0,float(y1-y0)/height,y0],[0,0,1]]
    matrix=np.asarray(matrix,np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
        raise ValueError('view plan must give a finite nondegenerate pixel-corner transform')
    yy,xx=np.mgrid[:height,:width]
    uv=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@matrix.T
    uv=uv[...,:2]/uv[...,2:]
    return (width,height),matrix,uv,plan


def _neighbors(uv, shape):
    xy=uv-.5
    safe=np.nan_to_num(xy,nan=-1e6,posinf=-1e6,neginf=-1e6)
    low=np.floor(safe).astype(np.int64)
    high=low+1
    inside=np.isfinite(xy).all(2)&(low[...,0]>=0)&(low[...,1]>=0)&(high[...,0]<shape[1])&(high[...,1]<shape[0])
    low=np.clip(low,[0,0],[shape[1]-1,shape[0]-1])
    high=np.clip(high,[0,0],[shape[1]-1,shape[0]-1])
    fraction=np.clip(safe-np.floor(safe),0,1)
    return low,high,fraction,inside


def _four(array, neighbors):
    low,high,_,_=neighbors
    x,y=low[...,0],low[...,1]
    x1,y1=high[...,0],high[...,1]
    return np.stack([array[y,x],array[y,x1],array[y1,x],array[y1,x1]],axis=-1)


def _sample(array, neighbors):
    low,high,fraction,inside=neighbors
    x,y=low[...,0],low[...,1]
    x1,y1=high[...,0],high[...,1]
    fx,fy=fraction[...,0],fraction[...,1]
    if array.ndim==3:fx,fy=fx[...,None],fy[...,None]
    value=((1-fx)*(1-fy)*array[y,x]+fx*(1-fy)*array[y,x1]
           +(1-fx)*fy*array[y1,x]+fx*fy*array[y1,x1])
    return value


def _paint_visible(rgb, visible):
    image=rgb.astype(np.float32)/255
    luminance=image@np.array([.2126,.7152,.0722],np.float32)
    median=float(np.median(luminance[visible])) if visible.any() else .3
    threshold=float(np.clip(median+.09,.35,.85))
    r,g,b=image.transpose(2,0,1)
    white=(luminance>threshold)&(np.ptp(image,axis=2)<.20)&visible
    yellow=(r-b>.055)&(g-b>.035)&(r-g<.22)&(luminance>.20)&visible
    return white,yellow


def render_view(layers, view_plan):
    """Render a chosen crop; all source roles use one pixel-center transform."""
    start=time.perf_counter()
    rgb=layers['rgb'];native=layers['masks'];neutral=layers['neutral_rgb']
    independent=layers['receipt'].get('independent_regions',{}).get('mode')=='independent'
    if independent:
        from .independent_regions import validate_independent_regions
        validate_independent_regions(native,rgb.shape[:2])
    wh,matrix,uv,plan=_plan(view_plan,rgb.shape[:2])
    neighbors=_neighbors(uv,rgb.shape[:2]);inside=neighbors[3]
    if plan.get('content_box_xyxy') is not None:
        x0,y0,x1,y1=plan['content_box_xyxy'];yy,xx=np.mgrid[:wh[1],:wh[0]]
        inside &= (xx+.5>=x0)&(xx+.5<x1)&(yy+.5>=y0)&(yy+.5<y1)
    reliable={'C_mat','road_observed','visible_support','depth_valid','geometry_observed_valid','geometry_road_visible','road_geometry_observed'}
    masks={name:(_four(value,neighbors).all(-1) if name in reliable else _four(value,neighbors).any(-1))&inside
           for name,value in native.items()}
    # Conservative footprints can touch after resampling even when the native
    # roles are disjoint. Protection wins on the final tensor grid; keep the
    # conflict as uncertainty rather than silently deleting a retained label.
    conflict=masks['target_unknown']&masks['K']
    masks['resampling_role_conflict']=conflict
    masks['uncertainty']|=conflict
    for name in ['target_unknown','rgb_unknown_core','physical_unknown']:
        masks[name]&=~masks['K']
    f=masks['target_unknown']
    cmat=masks['C_mat']&~f
    masks['C_mat']=cmat;masks['visible_support']=cmat
    masks['canvas_valid']=inside
    masks['retained_context']=~f&inside
    # New probability mode provides a separate, authorized native write region.
    # Legacy inputs have no native W and keep their original F-copy behavior.
    masks['W_edit']=(masks.get('W_edit',f)&f)&~masks['K']
    if independent:
        # Pending wins where conservative resampling footprints touch. Extra
        # hidden-only pixels never become unresolved output edits.
        pending=masks['pending_output']
        masks['W_edit'] &= ~pending
        masks.update(H_input=f.copy(),W_output=masks['W_edit'].copy(),
                     pending_output=pending.copy(),shadow_pending=pending.copy(),
                     input_only=f&~masks['W_edit']&~pending,M_material=~cmat,
                     remaining_effect=masks['remaining_effect']&~masks['W_edit'])
        validate_independent_regions(masks,(wh[1],wh[0]))
    elif 'shadow_pending' in native:
        # Keep unresolved ownership on K as a diagnostic; K still wins F/W.
        masks['shadow_pending'] |= f&~masks['W_edit']
    geo_source=rgb.copy();geo_source[native['target_unknown']]=neutral
    observed_source=rgb.copy();observed_source[native['rgb_unknown_core']]=neutral
    mat_source=rgb.copy();mat_source[~native['C_mat']]=neutral
    geo=np.rint(np.clip(_sample(geo_source,neighbors),0,255)).astype(np.uint8)
    material=np.rint(np.clip(_sample(mat_source,neighbors),0,255)).astype(np.uint8)
    geometry_observation=np.rint(np.clip(_sample(observed_source,neighbors),0,255)).astype(np.uint8)
    geo[f|~inside]=neutral;material[~cmat]=neutral
    # Preserve K from legal known context only. Normalizing native known-source
    # weights avoids mixing neutral core pixels into a protected object edge.
    # Neither unknown core nor removed photometric-effect RGB is consulted.
    known_native=~native['target_unknown']
    known_source=rgb.copy();known_source[~known_native]=0
    known_weight=_sample(known_native.astype(np.float32),neighbors)
    known_value=_sample(known_source,neighbors)/np.maximum(known_weight[...,None],1e-12)
    restore=masks['K']&inside&(known_weight>1e-12)
    geo[restore]=np.rint(np.clip(known_value[restore],0,255)).astype(np.uint8)
    masks['protected_context_unavailable']=masks['K']&inside&~restore
    geometry_observation[masks['rgb_unknown_core']|~inside]=neutral
    white,yellow=_paint_visible(material,cmat)
    masks['visible_white']=white;masks['visible_yellow']=yellow
    geometry_white,geometry_yellow=_paint_visible(geometry_observation,masks['geometry_road_visible'])
    masks['geometry_visible_white']=geometry_white;masks['geometry_visible_yellow']=geometry_yellow
    loss=_four(layers['loss_mask'],neighbors).all(-1)&inside
    target=None
    if layers['target_sanitized'] is not None:
        target=np.rint(np.clip(_sample(layers['target_sanitized'],neighbors),0,255)).astype(np.uint8)
        target[masks['foreground_possible']|masks['foreground_uncertainty']|~inside]=neutral
        clean_supervision=np.rint(np.clip(_sample(layers['supervision_rgb'],neighbors),0,255)).astype(np.uint8)
        target[loss]=clean_supervision[loss]
    confidence={name:_sample(value,neighbors).astype(np.float32) for name,value in layers['confidence'].items()}
    for value in confidence.values():value[~inside]=0
    depth=None
    if layers['depth_hint'] is not None:
        depth=_sample(layers['depth_hint'],neighbors).astype(np.float32)
        depth[~masks['depth_valid']]=0
    assert np.all(geo[f]==neutral) and np.all(material[~cmat]==neutral)
    assert not (cmat&(masks['foreground_possible']|f)).any()
    valid_pixels=material[cmat].astype(np.float32)/255
    features=dict(target_effective_pixels=int(masks['target_core'].sum()),unknown_pixels=int(f.sum()),
                  material_visible_pixels=int(cmat.sum()),material_fraction=float(cmat.mean()),
                  retained_geometry_pixels=int((masks['retained_core']&~f).sum()),
                  uncertainty_fraction=float(masks['uncertainty'].mean()),
                  visible_white_pixels=int(white.sum()),visible_yellow_pixels=int(yellow.sum()),
                  material_luma_median=float(np.median(valid_pixels@np.array([.2126,.7152,.0722]))) if len(valid_pixels) else 0.,
                  source_to_input_scale_xy=np.linalg.norm(matrix[:2,:2],axis=0).tolist())
    receipt=dict(schema='dual_context_input.v1',output_wh=list(wh),native_from_input=matrix.tolist(),
                 input_from_native=np.linalg.inv(matrix).tolist(),view_plan=_plain(plan),
                 features=features,loss_pixels=int(loss.sum()),
                 geometry_target_sanitized_before_resampling=True,
                 material_forbidden_sanitized_before_resampling=True,
                 material_source_four_neighbors_reliable=True,
                 geometry_other_objects_retained=True,accepted_masks_not_gated_by_current_lidar=True,
                 protected_resampling_policy='K priority; native known-context normalized bilinear',
                 resampling_role_conflict_pixels=int(conflict.sum()),
                 protected_context_unavailable_pixels=int(masks['protected_context_unavailable'].sum()),
                 hidden_paint_used_for_condition=False,real_hidden_road_gt=False,
                 render_seconds=time.perf_counter()-start)
    observation=np.rint(np.clip(_sample(rgb,neighbors),0,255)).astype(np.uint8)
    if 'object_shadow' in layers['receipt']:
        from .shadow_probability import rgb_sha256
        receipt['observation_sha256']=rgb_sha256(observation)
        receipt['object_shadow']=dict(layers['receipt']['object_shadow'],
            output_pending_pixels=int(masks['shadow_pending'].sum()),
            write_region_is_explicit=True,shadow_probabilities_preserved=True)
    if independent:
        from .shadow_probability import rgb_sha256
        receipt['observation_sha256']=rgb_sha256(observation)
        receipt['independent_regions']=dict(layers['receipt']['independent_regions'],
            output_H_input_pixels=int(masks['H_input'].sum()),
            output_W_output_pixels=int(masks['W_output'].sum()),
            output_input_only_pixels=int(masks['input_only'].sum()),
            output_pending_pixels=int(masks['pending_output'].sum()),
            output_remaining_effect_pixels=int(masks['remaining_effect'].sum()))
    return dict(I_geo=geo,I_mat=material,geometry_observation=geometry_observation,masks=masks,confidence=confidence,depth_hint=depth,
                observation_rgb=observation,
                target_sanitized=target,loss_mask=loss,native_from_input=matrix,receipt=receipt)


def build_inputs(rgb, road_mask, target_core, view_plan, **native_arguments):
    """One common convenience entry for pseudo-supervised and real queries."""
    return render_view(prepare_native_layers(rgb,road_mask,target_core,**native_arguments),view_plan)
