"""Repair the exact high-recall M1 candidate using frozen native alphaT moments."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch


def apply_recall(rows, source, output, frame, transform, stage):
    path=Path(source)/('selection_'+frame+'.npz')
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    verified_path=Path(source).parent/'recall_verified'/path.name
    with np.load(verified_path,allow_pickle=False) as data:
        if not np.array_equal(arrays['selected'],data['selected']):
            raise ValueError('Recall candidate changed between verified M1 and moment pass')
        arrays['body_alphaT']=data['body_alphaT'].copy()
        arrays['background_alphaT']=data['background_alphaT'].copy()
    # Targets are computed in the unedited train view, including wrong-depth contributors.
    selected=torch.as_tensor(arrays['selected'],device=rows['means3D'].device)
    body=torch.as_tensor(arrays['body_alphaT'],device=selected.device)
    bg=torch.as_tensor(arrays['background_alphaT'],device=selected.device)
    vt=torch.as_tensor(arrays['vehicle_surface_target_model'],device=selected.device,dtype=rows['means3D'].dtype)
    bt=torch.as_tensor(arrays['background_surface_target_model'],device=selected.device,dtype=rows['means3D'].dtype)
    valid_v=torch.as_tensor(arrays['vehicle_target_valid'],device=selected.device)&torch.isfinite(vt).all(1)&(body>1e-4)
    valid_b=torch.as_tensor(arrays['background_target_valid'],device=selected.device)&torch.isfinite(bt).all(1)&(bg>1e-4)
    repairable=selected&valid_v
    result=dict(rows)
    result['means3D']=rows['means3D'].clone()
    if stage != 'assignment':
        result['means3D'][repairable]=vt[repairable]
    scale=float(np.linalg.norm(np.asarray(transform)[:3,0]))
    displacement=(vt[repairable]-rows['means3D'][repairable]).norm(dim=1)/scale
    shrunk=0
    if stage in ('position_scale','position_scale_split'):
        result['scales']=rows['scales'].clone()
        original=rows['scales'][repairable]
        factor=(.08*scale/original.max(dim=1).values.clamp_min(1e-12)).clamp(max=1.)
        result['scales'][repairable]=original*factor[:,None]
        shrunk=int((factor<1).sum())
    parents=torch.empty(0,dtype=torch.long,device=selected.device)
    conservation=0.
    if stage=='position_scale_split':
        parents=torch.where(selected&valid_v&valid_b)[0]
        result={k:None if v is None else torch.cat([v,v[parents].clone()],dim=0) for k,v in result.items()}
        if len(parents):
            result['means3D'][-len(parents):]=bt[parents]
            p=(body[parents]/(body[parents]+bg[parents])).clamp(.01,.99)
            opacity=rows['opacities'][parents].clamp(0,1-1e-7)
            av=1-(1-opacity)**p.reshape_as(opacity)
            ab=1-(1-opacity)**(1-p).reshape_as(opacity)
            result['opacities'][parents]=av
            result['opacities'][-len(parents):]=ab
            conservation=float((1-(1-av)*(1-ab)-opacity).abs().max())
        selected=torch.cat([selected,torch.zeros(len(parents),dtype=torch.bool,device=selected.device)])
    output=Path(output)
    shutil.copy2(path,output/path.name)
    np.savez_compressed(output/('repaired_geometry_'+frame+'.npz'),means3D=result['means3D'].cpu().numpy(),scales=result['scales'].cpu().numpy(),opacities=result['opacities'].cpu().numpy(),selected=selected.cpu().numpy(),split_parent_indices=parents.cpu().numpy())
    receipt=dict(frame=frame,stage=stage,selected_original=int(arrays['selected'].sum()),repairable=int(repairable.sum()),unchanged_selected_without_valid_target=int((torch.as_tensor(arrays['selected'],device=selected.device)&~valid_v).sum()),
        shrunk=shrunk,split_parents=len(parents),capacity_added=len(parents),
        colocated_transmittance_max_abs_error=conservation,
        source_selection=str(path),source_selection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target='actual original unified native alphaT-weighted independent first-hit vehicle/nonvehicle XYZ moments; no depth rejection',
        split_trigger='same M1 recall selected AND original actual vehicle/nonvehicle alphaT each >1e-4 with finite target',
        split_fraction='body/(body+background) clipped [.01,.99]; optical thickness fractions',
        displacement_world_m_mean=float(displacement.mean()) if len(displacement) and stage!='assignment' else 0.,displacement_world_m_max=float(displacement.max()) if len(displacement) and stage!='assignment' else 0.,
        scale_cap_world_m=.08 if stage in ('position_scale','position_scale_split') else None,optimization_updates=0,
        frozen='unselected static, original dynamic, color/SH, rotations, environment; split inherits SH, rotation and capped parent scale',
        limitations='per-query truth-assisted intervention, no RGB fitting, no exact nearest-mesh target, relocation does not preserve per-pixel alpha; not mathematical upper bound')
    return result,selected,receipt


def replay(rows, path):
    with np.load(path,allow_pickle=False) as saved:
        parents=torch.as_tensor(saved['split_parent_indices'],device=rows['means3D'].device,dtype=torch.long)
        result={k:None if v is None else torch.cat([v,v[parents]],dim=0) for k,v in rows.items()}
        for key in ('means3D','scales','opacities'):
            result[key]=torch.as_tensor(saved[key],device=rows['means3D'].device)
        selected=torch.as_tensor(saved['selected'],device=rows['means3D'].device)
        if 'refit_indices' in saved:
            indices=torch.as_tensor(saved['refit_indices'],device=rows['means3D'].device,dtype=torch.long)
            result['shs']=result['shs'].clone()
            result['shs'][indices,0,:]=torch.as_tensor(saved['refit_dc'],device=rows['means3D'].device)
    return result,selected,dict(replay_geometry=str(path),heldout_GT_used_for_repair=False)
