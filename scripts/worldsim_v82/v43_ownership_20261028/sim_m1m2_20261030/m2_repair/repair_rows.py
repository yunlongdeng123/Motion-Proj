"""Finite-support truth-assisted representation intervention; no learned update."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def repair_rows(rows, selected, frame, reference_root, transform, stage, selection_path):
    if stage not in ('position', 'position_scale'):
        raise ValueError(stage)
    selection_path = Path(selection_path)
    with np.load(selection_path, allow_pickle=False) as data:
        saved = data['selected']
        target_np = data['vehicle_surface_target_model'].copy()
        support = data['vehicle_support'].copy()
    if not np.array_equal(saved, selected.detach().cpu().numpy()):
        raise ValueError('M1/M2 candidate selection differs')
    if np.any(support[saved] == 0) or not np.isfinite(target_np[saved]).all():
        raise ValueError('Selected positions require finite observed vehicle support')
    scale = float(np.linalg.norm(np.asarray(transform)[:3, 0]))
    target = torch.as_tensor(target_np, device=selected.device, dtype=rows['means3D'].dtype)
    result = dict(rows)
    result['means3D'] = rows['means3D'].clone()
    result['means3D'][selected] = target[selected]
    displacement = (target[selected] - rows['means3D'][selected]).norm(dim=-1) / scale
    shrunk = 0
    if stage == 'position_scale':
        result['scales'] = rows['scales'].clone()
        original = rows['scales'][selected]
        factor = (0.08 * scale / original.max(dim=1).values.clamp_min(1e-12)).clamp(max=1.)
        result['scales'][selected] = original * factor[:, None]
        shrunk = int((factor < 1).sum())
    # All other arrays are shared, never mutated. Nonselected rows remain bitwise equal.
    for key, value in rows.items():
        if key not in ('means3D', 'scales'):
            assert result[key] is value
        elif value is not None:
            assert torch.equal(result[key][~selected], value[~selected])
    selected_indices = np.flatnonzero(saved).astype(np.int64)
    receipt = dict(stage=stage, frame=frame, selected=int(selected.sum()), shrunk=shrunk,
        selected_index_sha256=hashlib.sha256(selected_indices.tobytes()).hexdigest(),
        source_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        position_target='centroid of M1 accepted independent first-hit vehicle surface support points; not exact nearest mesh projection',
        displacement_world_m_mean=float(displacement.mean()) if len(displacement) else 0.,
        displacement_world_m_max=float(displacement.max()) if len(displacement) else 0.,
        scale_cap_world_m=0.08 if stage == 'position_scale' else None,
        capacity_added=0, split_performed=False, optimization_updates=0,
        frozen='SH/color, opacity, quaternion, learned dynamic branch, environment, all nonselected static rows',
        privileged_inputs='M1 finite covariance support depth/semantic associations and calibrated first-hit surface coordinates; no GT RGB copied',
        conditional_reference=True, mathematical_upper_bound=False,
        limitation='Per-query finite-support intervention, not persistent camera-independent Gaussian reparameterization; missing/unmatched support remains unrepaired')
    return result, receipt


def split_mixed(rows, selected, camera, transform, reference_root, frame, selection_path, alpha_path):
    """Split only selected mixed support confirmed by actual compositing contributions."""
    from utils.general_utils import build_rotation
    with np.load(selection_path, allow_pickle=False) as data:
        mixed = torch.tensor(data['category'] == 1, device=selected.device)
        means = torch.tensor(data['means_model'], device=selected.device)
        scales = torch.tensor(data['scales_model'], device=selected.device)
        rotations = torch.tensor(data['rotations'], device=selected.device)
    with np.load(alpha_path, allow_pickle=False) as data:
        body = torch.tensor(data['body_alphaT'], device=selected.device)
        background = torch.tensor(data['background_alphaT'], device=selected.device)
    trigger = selected & mixed & (body > 1e-4) & (background > 1e-4)
    indices = torch.where(trigger)[0]
    if not len(indices):
        return rows, selected, dict(split_parents=0, capacity_added=0, trigger='selected mixed finite support AND actual body/background alphaT >1e-4')
    scale=float(np.linalg.norm(np.asarray(transform)[:3,0]))
    with np.load(Path(reference_root)/'original'/(frame+'.npz')) as data:
        depth=torch.tensor(data['depth_z_m'],device=selected.device,dtype=means.dtype)
        semantic=torch.tensor(data['semantic_id'].astype(np.int32),device=selected.device)
    h,w=depth.shape
    axes=build_rotation(rotations[indices])*scales[indices,None,:]
    offsets=[torch.zeros_like(means[indices])]
    for radius in (1.,2.):
        for axis in range(3):
            offsets.extend([axes[:,:,axis]*radius,-axes[:,:,axis]*radius])
    sums=torch.zeros_like(means[indices]); counts=torch.zeros(len(indices),device=selected.device)
    for offset in offsets:
        pts=means[indices]+offset
        cam=torch.cat([pts,torch.ones_like(pts[:,:1])],1)@camera.world_view_transform
        z=cam[:,2]; u=torch.floor(camera.fx*cam[:,0]/z+camera.cx).long(); v=torch.floor(camera.fy*cam[:,1]/z+camera.cy).long()
        inside=(z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
        uu=u.clamp(0,w-1); vv=v.clamp(0,h-1); d=depth[vv,uu]
        accept=inside&(d>0)&(semantic[vv,uu]!=10)&((z/scale-d).abs()<=.35)
        pcam=torch.stack(((uu+.5-camera.cx)/camera.fx*d*scale,(vv+.5-camera.cy)/camera.fy*d*scale,d*scale,torch.ones_like(d)),1)
        hits=(pcam@torch.linalg.inv(camera.world_view_transform))[:,:3]
        sums+=hits*accept[:,None]; counts+=accept
    valid=counts>0; indices=indices[valid]; target=sums[valid]/counts[valid,None]
    result={key:(None if value is None else torch.cat([value,value[indices].clone()],dim=0)) for key,value in rows.items()}
    result['means3D'][-len(indices):]=target
    # Conserves optical thickness for co-located equal-footprint children, not pixel alpha after relocation.
    p=(body[indices]/(body[indices]+background[indices])).clamp(.01,.99)
    opacity=rows['opacities'][indices].clamp(0,1-1e-7)
    ov=1-torch.pow(1-opacity,p.reshape_as(opacity))
    ob=1-torch.pow(1-opacity,(1-p).reshape_as(opacity))
    result['opacities'][indices]=ov
    result['opacities'][-len(indices):]=ob
    conservation=(1-(1-ov)*(1-ob)-opacity).abs().max()
    new_selected=torch.cat([selected,torch.zeros(len(indices),dtype=torch.bool,device=selected.device)])
    return result,new_selected,dict(split_parents=len(indices),capacity_added=len(indices),
        parent_static_indices=indices.cpu().tolist(),trigger='M1 selected mixed sigma surface support AND actual body/background alphaT >1e-4',
        opacity_rule='optical thickness fractions by original actual body/background alphaT; fractions clipped [.01,.99]',
        colocated_transmittance_max_abs_error=float(conservation),
        opacity_not_globally_conserved_after_child_relocation=True,background_child_target='accepted independent nonvehicle first-hit support centroid',
        child_colors='both inherit original parent SH; no GT RGB',optimization_updates=0)
