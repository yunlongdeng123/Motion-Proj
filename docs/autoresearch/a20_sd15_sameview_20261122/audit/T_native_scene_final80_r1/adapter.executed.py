"""Frozen 410080-row T asset in the unchanged native scene; CPU default.

Uses the already-frozen mode-aware old-row masks. No row selection, optimizer,
new-Gaussian clipping, pose fitting or scene modification is available here.
Activated opacity (including 1.0) is passed directly; it is never logit-clamped.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import numpy as np

BUILD=(86,110)
TRACK='bzFbWRrSGOgEaXxnuGV0tw'
BASE_T_SHA='385001b9a5a49a376a809ea7f1b49809d8a6eaa72d6d913d6ffd241e44287c9f'
SELECTION_SHA='de59356129729b3262fe161d6cb51d579da9c3e76cd736c4e2ff337da706108f'
ROAD_PROOF_SHA='684973c593b6064b9f6d1c1e0537fb2476307962d9c8ec26d3a853ac555921ec'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def ref(p):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def packet(a):
    q=a['rotation_wxyz'] if 'rotation_wxyz' in a else a['rotation']
    if 'features_dc' in a:dc=np.asarray(a['features_dc']).reshape(-1,3)
    elif 'coefficients' in a:
        if a['coefficients'].shape[1:]!=(16,3) or np.any(a['coefficients'][:,1:]!=0):
            raise ValueError('This fixed control may contain DC only; higher SH must remain zero')
        dc=a['coefficients'][:,0]
    else:raise ValueError('Expected activated features_dc or normalized DC coefficients')
    out=dict(xyz=np.asarray(a['xyz']),scales=np.asarray(a['scales']),rotation=np.asarray(q),
        opacity=np.asarray(a['opacity']).reshape(-1),dc=np.asarray(dc))
    n=len(out['xyz'])
    shapes=dict(xyz=(n,3),scales=(n,3),rotation=(n,4),opacity=(n,),dc=(n,3))
    if n!=410080:raise ValueError('The support-preserving T diagnostic requires exactly410080 rows')
    for key,shape in shapes.items():
        if out[key].shape!=shape or out[key].dtype!=np.float32 or not np.isfinite(out[key]).all():
            raise ValueError('Expected finite exact float32 activated field: '+key)
    if (out['scales']<=0).any() or ((out['opacity']<0)|(out['opacity']>1)).any():raise ValueError('Invalid activated parameters')
    if np.max(np.abs(np.linalg.norm(out['rotation'],axis=1)-1))>1e-4:raise ValueError('Non-unit activated quaternion')
    return out


def check(args):
    if sha(args.baseline_T)!=BASE_T_SHA:raise ValueError('Original registered same-SLAT T baseline changed')
    if sha(args.selection_cache/'native21_contribution_selection.npz')!=SELECTION_SHA:
        raise ValueError('Frozen mode-aware row masks changed')
    source=bank(args.asset);a=packet(source);base=packet(bank(args.baseline_T))
    geometry={k:bool(np.array_equal(a[k],base[k]) and array_sha(a[k])==array_sha(base[k])) for k in ['xyz','scales','rotation','opacity']}
    if not all(geometry.values()):raise ValueError('Diagnostic must retain original registered T geometry/covariance/opacity exactly')
    changed=(a['dc']!=base['dc']).any(1);appearance_ref=None;mask_key=None
    if args.appearance_mask:
        mask_source=bank(args.appearance_mask);mask_key=args.appearance_mask_key
        if mask_key not in mask_source:raise ValueError('Explicit appearance support key missing')
        allowed=np.asarray(mask_source[mask_key],bool)
        appearance_ref=ref(args.appearance_mask)
    elif 'appearance_trainable' in source:
        allowed=np.asarray(source['appearance_trainable'],bool);mask_key='appearance_trainable'
        appearance_ref=ref(args.asset)
    elif not changed.any():allowed=np.zeros(len(changed),bool)
    else:raise ValueError('DC-changed control requires its frozen true-V appearance support mask')
    if allowed.shape!=(410080,) or (changed&~allowed).any():raise ValueError('DC change outside frozen V appearance support')
    if array_sha(a['dc'][~allowed])!=array_sha(base['dc'][~allowed]):raise ValueError('U DC bytes changed')
    if args.asset_kind=='initial_T' and changed.any():raise ValueError('Initial T diagnostic may not change DC')
    previous=json.loads((args.selection_cache/'grey_renderer_receipt.json').read_text())
    for name,path in [('legacy',args.legacy),('sidecar',args.sidecar)]:
        if previous['ready_contract'][name]['sha256']!=sha(path):raise ValueError('Native scene binding changed: '+name)
    if sha(args.road_source/'persistent/proof.json')!=ROAD_PROOF_SHA:raise ValueError('Frozen road/native scene proof changed')
    mask_bindings={}
    for raw in BUILD:
        frozen=previous['selection']['source_masks'][str(raw)]
        labels=Path(frozen['labels']['path']);visible=args.conditioning/'train'/f'{raw:03d}'/'visible_mask.png'
        if sha(labels)!=frozen['labels']['sha256'] or sha(visible)!=frozen['V']['sha256']:
            raise ValueError('Frozen K/V metric masks changed')
        mask_bindings[str(raw)]=dict(labels=ref(labels),V=ref(visible))
    selection=bank(args.selection_cache/'native21_contribution_selection.npz')
    if int(selection['retire_KEEP'].sum())!=14994 or int(selection['retire_DELETE'].sum())!=24449:
        raise ValueError('Unexpected frozen retired row counts')
    float_refs={}
    for raw in BUILD:
        for mode in ['keep','delete']:
            p=args.selection_cache/'renders'/f'{raw:03d}'/('native_'+mode)/'render_float.npz'
            z=bank(p)
            for k in ['rgb','target_alpha','retained_native_K_alpha','retained_native11_alpha']:
                if k not in z:raise ValueError('Cached native baseline misses '+k)
            float_refs[f'{raw:03d}_{mode}']=ref(p)
    return dict(schema='a20.T.frozen_native_scene.cpu_preflight.v1',passed=True,
        adapter_source=ref(__file__),
        asset_kind=args.asset_kind,asset=ref(args.asset),baseline_T=ref(args.baseline_T),asset_receipt=None if not args.asset_receipt else ref(args.asset_receipt),
        activated_field_hashes={k:array_sha(v) for k,v in a.items()},geometry_exactly_equal=geometry,
        opacity_range=[float(a['opacity'].min()),float(a['opacity'].max())],activated_opacity_passed_directly=True,
        changed_DC_rows=int(changed.sum()),appearance_allowed_rows=int(allowed.sum()),appearance_support=appearance_ref,
        appearance_support_key=mask_key,U_DC_changed_rows=int((changed&~allowed).sum()),higher_SH_zero=True,
        selection=ref(args.selection_cache/'native21_contribution_selection.npz'),
        frozen_K_V_metric_masks=mask_bindings,
        frozen_mode_retire_counts=dict(KEEP=14994,DELETE=24449),selection_rerun=False,
        native_float_baselines=float_refs,legacy=ref(args.legacy),sidecar=ref(args.sidecar),
        road_proof=ref(args.road_source/'persistent/proof.json'),
        BUILD=list(BUILD),heldout_frames_opened=[],planned_rasters=6,model_parameter_updates=0,optimizer_available=False,
        proper_registered_coordinates_reused=True,physical_contact_transform_applied=False,
        DC_only_update_cannot_change_K_alpha_if_geometry_is_equal=True,
        no_per_camera_new_Gaussian_clipping=True,full_scene_RGB_not_used_for_training=True,
        scene_promotion_allowed=False,gpu_used=False,torch_imported='torch' in sys.modules)


def preview(args,ready):
    if not args.authorize_final_asset_diagnostic:raise ValueError('Root must supply the final frozen asset and queue the GPU diagnostic')
    if args.output.exists():raise ValueError('Use a fresh frozen diagnostic output directory')
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'entry_contract.json',ready)
    spec=importlib.util.spec_from_file_location('_frozen_T_native_legacy',args.legacy)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    if (m.TARGET,m.DELETE,m.TRACK)!=(21,10,TRACK):raise ValueError('Native actor identity mismatch')
    m.ROOT=args.output;m.FRAMES=BUILD;m.TRAIN=BUILD
    torch=m.torch
    if not torch.cuda.is_available():raise RuntimeError('Final diagnostic requires actual CUDA')
    torch.set_grad_enabled(False);torch.cuda.reset_peak_memory_stats();started=time.monotonic()

    class FrozenT(torch.nn.Module):
        def __init__(self,a):
            super().__init__()
            for key,value in packet(a).items():self.register_buffer('base_'+key,m.ten(value))
        def rows(self,pose,cam):
            R=pose[:3,:3]/pose[:3,:3].norm(dim=0).mean();scale=pose[:3,:3].norm(dim=0).mean()
            xyz=self.base_xyz@pose[:3,:3].T+pose[:3,3]
            color=(self.base_dc*.28209479177387814+.5).clamp_min(0)
            qp=m.ten(m.quat_from_matrix(R.detach().cpu().numpy()[None]))[0]
            rotation=m.qmul(qp.expand_as(self.base_rotation),self.base_rotation)
            depth=xyz@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2]
            f=xyz.new_zeros((len(xyz),24));f[:,0]=depth;f[:,1]=1;f[:,2]=1
            return dict(means3D=xyz,means2D=torch.zeros_like(xyz),colors_precomp=color,shs=None,features=f,
                opacities=self.base_opacity[:,None],scales=self.base_scales*scale,rotations=rotation,
                cov3D_precomp=None,mask=depth>.00377)
    m.Vehicle=FrozenT
    class Experiment(m.Experiment):
        def build_vehicle(self,path):
            shutil.copy2(args.asset,path)
            if sha(path)!=ready['asset']['sha256']:raise ValueError('Frozen T source copy changed')
        def build_domains(self):return
    e=Experiment(args.road_source);e.road_color.requires_grad_(False)
    for model in [e.scene.static,e.scene.dynamic,e.scene.environment]:
        if hasattr(model,'parameters'):
            for p in model.parameters():p.requires_grad_(False)
    select=bank(args.selection_cache/'native21_contribution_selection.npz')
    global_rows=m.ten(select['global_row_ids'],torch.long)
    retire={mode:global_rows[m.ten(select['retire_'+mode.upper()],torch.bool)] for mode in ['keep','delete']}
    side=bank(args.sidecar);guard=m.ten(side['global_row_ids'],torch.long)
    contexts={};partitions=[];seen=[]
    for raw in BUILD:
        packet_native,owners,ri=e.native_packets[raw]
        values,_,_,_=e.scene._rows(raw,(21,),(),False);ids=torch.nonzero(values['mask']).reshape(-1)
        if not torch.equal(values['means3D'][ids],packet_native['means3D']):raise ValueError('Native global row mapping changed')
        own=owners.clone();match=torch.isin(ids,guard)
        if ((own[match]>0)&(own[match]!=10)).any():raise ValueError('a9 fixed guard would alter retained actor')
        own[match]=10;seen.append(ids[own==21])
        for mode in ['keep','delete']:
            remove=(own==21)&torch.isin(ids,retire[mode]);keep=~remove
            p=m.subset(packet_native,keep);o=own[keep].clone();features=p['features'].clone();features[:,2:]=0
            features[:,3]=(ri[keep]>=0).float();legacy=o==21
            features[:,4]=legacy.float();features[:,5]=(o==10).float()
            features[:,8]=((o>0)&(o!=21)&(o!=10)).float();features[:,9]=(o==11).float();o[legacy]=0
            contexts[(raw,mode)]=(dict(p,features=features),o,ri[keep])
            partitions.append(dict(raw=raw,mode=mode,retired_rows=int(remove.sum()),legacy_native21_rows=int(legacy.sum()),
                selection_reused=True,original_row_parameters_unchanged=True))
    if not torch.equal(torch.unique(torch.cat(seen)),global_rows):raise ValueError('Persistent native21 identity union changed')

    def render(raw,mode='keep',isolated=False):
        car=e.vehicle.rows(e.poses[raw],e.cameras[raw])
        if isolated:rows=car
        else:
            p,o,ri=contexts[(raw,mode)];road_ids=torch.nonzero(ri>=0).reshape(-1)
            p=dict(p,colors_precomp=p['colors_precomp'].index_copy(0,road_ids,e.colors()[ri[road_ids]]))
            rows=e.scene._bindings().merge_rows(p,car)
            owners=torch.cat([o,torch.full((len(car['means3D']),),21,device='cuda',dtype=torch.long)])
            if mode=='delete':rows=dict(rows,mask=rows['mask']&(owners!=10))
        out=e.scene._bindings().raster(e.cameras[raw],e.scene.pipe,0,rows,e.scene._background);e.native_calls+=1
        rgb=out['rgb'] if isolated else out['rgb']+(1-out['alpha'])*e.env[raw]
        alpha=out['alpha'].squeeze();depth=out['depth_sum'].squeeze()/alpha.clamp_min(.90)/e.scale
        return dict(rgb=rgb,alpha=alpha,depth_camera_m=depth,target_alpha=out['extra'][0],road_alpha=out['extra'][1],
            legacy_native21_alpha=out['extra'][2],native_a9_alpha=out['extra'][3],
            retained_native_K_alpha=out['extra'][6],retained_native11_alpha=out['extra'][7])

    def potential(raw):
        from PIL import Image
        folder=args.conditioning/'train'/f'{raw:03d}';mapping=json.loads((folder/'mapping.json').read_text())
        w,h=mapping['image_native_size'];x0,y0,x1,y1=mapping['crop_native_xyxy_half_open']
        full=np.zeros((h,w),np.uint8)
        full[y0:y1,x0:x1]=((np.asarray(Image.open(folder/'visible_mask.png'))>0)|(np.asarray(Image.open(folder/'unknown_potential_mask.png'))>0))
        return m.ten(np.asarray(Image.fromarray(full).resize((1152,768),Image.Resampling.NEAREST))>0,torch.bool)

    metrics=[];alpha_refs={};a9_rows=[]
    with torch.inference_mode():
        for raw in BUILD:
            isolated=render(raw,isolated=True);folder=args.output/'renders'/f'{raw:03d}'/'isolated_T'
            for name,x in isolated.items():m.png(folder/(name+'.png'),x if name!='depth_camera_m' else (x/50).clamp(0,1))
            np.savez_compressed(folder/'render_float.npz',**{k:m.cpu(v).numpy() for k,v in isolated.items()})
            alpha_refs[str(raw)]=ref(folder/'render_float.npz')
            labels=e.targets[raw]['labels'];outside=~potential(raw);a9=labels==10
            for mode in ['keep','delete']:
                baseline=bank(args.selection_cache/'renders'/f'{raw:03d}'/('native_'+mode)/'render_float.npz')
                out=render(raw,mode);folder=args.output/'renders'/f'{raw:03d}'/('T_context_'+mode)
                for name,x in out.items():m.png(folder/(name+'.png'),x if name!='depth_camera_m' else (x/50).clamp(0,1))
                np.savez_compressed(folder/'render_float.npz',**{k:m.cpu(v).numpy() for k,v in out.items()})
                base={k:m.ten(v) for k,v in baseline.items()};delta=(out['rgb']-base['rgb']).abs();pixel=delta.amax(0)
                for name,mask,feature in [('all_retained_actors_K',(labels>0)&(labels!=21)&(labels!=10),'retained_native_K_alpha'),
                    ('native11_right_white_pickup',labels==11,'retained_native11_alpha')]:
                    change=(out[feature]-base[feature]).abs();n=int(mask.sum())
                    metrics.append(dict(raw=raw,mode=mode,K_group=name,pixels=n,
                        RGB_max_change=float(delta[:,mask].max()) if n else None,RGB_mean_change=float(delta[:,mask].mean()) if n else None,
                        RGB_pixels_changed_above_1_over_255=int(((pixel>1/255)&mask).sum()),
                        native_K_alpha_max_change=float(change[mask].max()) if n else None,
                        native_K_alpha_mean_change=float(change[mask].mean()) if n else None,
                        native_K_alpha_pixels_changed_above_1_over_255=int(((change>1/255)&mask).sum()),
                        new_T_shared_alpha_on_K_max=float(out['target_alpha'][mask].max()) if n else None,
                        strict_outside_potential_RGB_changed_pixels=int(((pixel>1/255)&outside).sum()),
                        strict_outside_potential_RGB_max=float(delta[:,outside].max())))
                a9_rows.append(dict(raw=raw,mode=mode,original_a9_mask_pixels=int(a9.sum()),
                    native_a9_alpha_in_original_a9_max=float(out['native_a9_alpha'][a9].max()),
                    legacy_native21_alpha_in_original_a9_max=float(out['legacy_native21_alpha'][a9].max()),
                    new_T_alpha_in_original_a9_max=float(out['target_alpha'][a9].max()),
                    legacy_overlap_does_not_establish_semantic_a9_residue=True))
    torch.cuda.synchronize()
    result=dict(schema='a20.T.frozen_native_scene.diagnostic.v1',entry_contract=ready,source_partition=partitions,
        K_RGB_alpha_metrics=metrics,a9_and_legacy_context=a9_rows,isolated_float_assets=alpha_refs,
        unchanged_T_geometry_makes_DC_only_update_alpha_invariant=True,
        activated_opacity_passed_directly=True,no_pose_or_road_adjustment=True,
        road_contact_evidence=ref(args.road_contact_evidence),road_contact_still_separate=True,
        native_renderer_calls=e.native_calls,model_parameter_updates=0,model_parameter_backward_calls=0,
        feature_VJP_calls=0,row_masks_reselected=False,new_Gaussians_clipped=False,RGB_mask_paste=False,
        GPU_stage_wall_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),scene_promotion_allowed=False,
        heldout_frames_opened=[],native_source_model_saw_full_sequence=True)
    write(args.output/'T_native_scene_receipt.json',result)
    if sha(args.asset)!=ready['asset']['sha256']:raise ValueError('Frozen control asset changed during diagnostic')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['cpu-check','preview'],default='cpu-check')
    p.add_argument('--asset-kind',choices=['initial_T','frozen_80V_control'],default='initial_T')
    for key in ['asset','baseline-T','selection-cache','legacy','sidecar','road-source','conditioning','road-contact-evidence','output']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--asset-receipt',type=Path);p.add_argument('--appearance-mask',type=Path)
    p.add_argument('--appearance-mask-key',default='appearance_trainable')
    p.add_argument('--authorize-final-asset-diagnostic',action='store_true')
    args=p.parse_args()
    if args.phase=='cpu-check':os.environ['CUDA_VISIBLE_DEVICES']=''
    ready=check(args)
    if args.phase=='cpu-check':write(args.output/'cpu_check.json',ready);print(json.dumps(ready,indent=2))
    else:preview(args,ready)
