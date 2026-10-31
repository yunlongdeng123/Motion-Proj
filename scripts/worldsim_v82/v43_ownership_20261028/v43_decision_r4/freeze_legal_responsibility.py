"""只由原native none和合法body mask冻结新H0；不读独立编辑参考。"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ.setdefault(k,'1')
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
from native_shared_route import NativeScene,read,write,file_hash
from r4_core import canonical_hash,enumerate_routes,freeze_family,register_tasks
from legal_mask import read_body_mask,validate_RGBA_equivalence


def legal_body_mask(native,frame):
    contract=Path(native.context['role']['original_input_contract']);data=read(contract)
    if native.context['role']['role']=='simulator_sensor_oracle':
        row=next(r for r in data['frames'] if str(r['id']).zfill(6)==frame)
        path=contract.parent/row['body_mask'];array,decode=read_body_mask(path)
        prep=read(native.scene_root/'preparation_manifest.json')
        expected=next(r for r in prep['frames'] if r['id']==frame)['body_pixels']
        if int(array.sum())!=int(expected):raise ValueError('正确RGB mask与原训练body_pixels分母不一致')
        return array,path,'inherited_simulator_sensor_oracle_original_body_mask',decode
    row=next(r for r in data['frames'] if r['frame_id']==frame)
    path=contract.parent/('%03d.npz'%row['raw'])
    with np.load(path,allow_pickle=False) as values:array=values['body'].copy().astype(bool)
    return array,path,'reviewed_original_DEVA_association_body_mask',dict(mode='boolean_NPZ',foreground_pixels=int(array.sum()),alpha_ignored=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-scene-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    contract=dict(schema='r4_legal_none_responsibility_domain_v1',parent_cpu_scene=str(args.cpu_scene_dir),
        mask_decoding='RGB foreground only; alpha ignored; compare original preparation body pixel counts',
        RGBA_decode_test=validate_RGBA_equivalence(),
        parent_sim_U_status='legacy_RGBA_decoder_used_all_source_support; preserve broad U ledger, not claimed valid target-body-neighbourhood',
        legal_evidence_only=True,reference_files_forbidden=True,operation='original_none_only',
        anchors='same_first_middle_last_as_parent',keep_domain='exact_complement_of_legal_body_mask_including_sky',
        contribution='native unified depth-sort alphaT from feature-channel1 adjoint; static and dynamic included in rendering',
        ratio='sum_anchor_body_alphaT/(sum_anchor_body_alphaT+sum_anchor_keep_alphaT)',
        minimum_total_alphaT_strict=1e-4,minimum_body_ratio_inclusive=.95,
        scope='only original parent U; intersect each original group, retain original anchor pose, empty groups omitted',
        thresholds_role='explicit_development_candidate_scope_not_calibrated_P_or_ownership_probability',
        original_representation_and_queries_unchanged=True,human_verdict=None)
    write(args.output/'pre_score_contract.json',contract)
    native=NativeScene(args.cpu_scene_dir,args.output/'restore')
    n=native.candidate['static_row_count'];body_sum=np.zeros(n,np.float64);keep_sum=np.zeros(n,np.float64)
    parent_u=np.zeros(n,bool)
    for g in native.candidate['groups']:parent_u[g['original_static_rows']]=True
    receipts=[];mask_refs=[]
    for frame in native.anchors:
        camera,static,dynamic,env=native.frame_rows(frame)
        mask,path,role,decode=legal_body_mask(native,frame);mask_refs.append(str(path))
        if mask.shape!=(camera.image_height,camera.image_width):raise ValueError('合法body mask与原native camera shape不符')
        body=torch.tensor(mask,dtype=torch.bool,device='cuda')
        combined=native.helpers.merge_rows(static,dynamic)
        with torch.enable_grad():
            features=combined['features'].detach().clone().requires_grad_(True)
            values=dict(combined,features=features)
            package=native.helpers.raster(camera,native.pipe,native.static.active_sh_degree,values,native.base_background)
            loss_body=(package['alpha'][0]*body).sum();loss_keep=(package['alpha'][0]*(~body)).sum()
            gb=torch.autograd.grad(loss_body,features,retain_graph=True)[0][:,1].detach()
            gk=torch.autograd.grad(loss_keep,features)[0][:,1].detach()
        if not torch.isfinite(gb).all() or not torch.isfinite(gk).all() or float(gb.min())< -1e-6 or float(gk.min())< -1e-6:
            raise ValueError('native alphaT责任非有限/负值')
        body_total=float(gb.double().sum());keep_total=float(gk.double().sum())
        body_native=float(loss_body.detach());keep_native=float(loss_keep.detach())
        if abs(body_total-body_native)>.05 or abs(keep_total-keep_native)>.05:raise ValueError('alphaT adjoint守恒检查失败')
        bs=gb[:n].cpu().numpy().astype(np.float64);ks=gk[:n].cpu().numpy().astype(np.float64)
        body_sum+=bs;keep_sum+=ks
        folder=args.output/'legal_responsibility';folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(frame+'.npz'),original_static_row=np.arange(n,dtype=np.int64),
            body_alphaT=bs,keep_alphaT=ks,parent_U=parent_u)
        receipts.append(dict(frame=frame,legal_body_mask=str(path),mask_sha256=file_hash(path),mask_role=role,
            body_pixel_denominator=int(mask.sum()),keep_pixel_denominator=int((~mask).sum()),
            mask_decoding=decode,
            native_static_rows=n,native_dynamic_prepared_rows=len(dynamic['means3D']),
            all_rows_body_alphaT=body_total,body_raster_alpha=body_native,
            all_rows_keep_alphaT=keep_total,keep_raster_alpha=keep_native,
            body_conservation_abs=abs(body_total-body_native),keep_conservation_abs=abs(keep_total-keep_native),
            independent_reference_read=False,renderer_forward_calls=1,feature_adjoint_backward_calls=2))
        print(json.dumps(dict(frame=frame,body_conservation_abs=abs(body_total-body_native),keep_conservation_abs=abs(keep_total-keep_native))),flush=True)
        del package,features,combined,values,gb,gk
    total=body_sum+keep_sum;ratio=np.divide(body_sum,total,out=np.zeros_like(total),where=total>0)
    selected=parent_u&(total>1e-4)&(ratio>=.95)
    candidate=deepcopy(native.candidate);groups=[];group_counts=[]
    for group in candidate['groups']:
        original=group['original_static_rows'];rows=[r for r in original if selected[r]]
        group_counts.append(dict(group_id=group['group_id'],parent_rows=len(original),operable_rows=len(rows),
            fixed_keep_rows=len(original)-len(rows),original_anchor=group['anchor_frame_id']))
        if rows:
            group['original_static_rows']=rows
            group['selection_ref']='legal_none_alphaT_body_ratio095';groups.append(group)
    candidate['groups']=groups
    candidate['parent_candidate_hash']=canonical_hash(native.candidate)
    candidate['parent_U_row_denominator']=int(parent_u.sum())
    candidate['legal_refinement']=dict(contract=contract,contract_hash=canonical_hash(contract),
        group_row_counts=group_counts,zero_support_parent_rows=int((parent_u&(total<=1e-4)).sum()),
        low_ratio_parent_rows=int((parent_u&(total>1e-4)&(ratio<.95)).sum()),
        all_nonselected_parent_rows_remain_fixed_keep=True)
    evidence=read(args.cpu_scene_dir/'legal_preedit_evidence.json')
    corrected_context=read(args.cpu_scene_dir/'camera_track_context.json')
    corrected_body_denominator=0
    for row in corrected_context['frames']:
        actual,_,_,_=legal_body_mask(native,row['frame_id'])
        row['original_body_pixels']=int(actual.sum());corrected_body_denominator+=int(actual.sum())
    evidence['metrics']['f_geom']['denominator']=corrected_body_denominator
    evidence['metrics']['f_geom']['measured_body_depth_pixel_denominator']=None
    evidence['metrics']['f_geom']['depth_denominator_status']='legacy_RGBA_derived_denominator_invalidated_recompute_pending'
    evidence['legal_observation_ids']=evidence['legal_observation_ids']+mask_refs
    evidence['legal_contribution_scope']=dict(operation='original_none',body_mask_refs=mask_refs,
        purity_threshold=.95,minimum_total_alphaT=1e-4,interpretation='observed_contribution_proxy_not_identity_or_P_qualification')
    family=freeze_family(candidate);routes=enumerate_routes(candidate)
    tasks=register_tasks(native.scene_id,candidate,evidence,routes,candidate['checkpoint_static_sha256'],native.frozen)
    frozen_root=args.output/'frozen_scene';frozen_root.mkdir();scene=frozen_root/native.scene_id;scene.mkdir()
    original_contract=read(args.cpu_scene_dir.parent/'run_contract.json')
    write(frozen_root/'run_contract.json',dict(original_contract,refinement_contract=contract))
    shutil.copy2(args.cpu_scene_dir/'checkpoint_cpu_receipt.json',scene/'checkpoint_cpu_receipt.json')
    write(scene/'camera_track_context.json',corrected_context)
    for name,value in [('candidate_manifest.json',candidate),('legal_preedit_evidence.json',evidence),
            ('frozen_H0.json',family),('shared_routes.json',routes),('tasks.json',tasks)]:write(scene/name,value)
    np.savez_compressed(args.output/'full_parent_U_ledger.npz',original_static_row=np.arange(n,dtype=np.int64),
        parent_U=parent_u,body_alphaT_sum=body_sum,keep_alphaT_sum=keep_sum,body_ratio=ratio,operable_in_new_H0=selected,
        fixed_keep_parent_row=parent_u&~selected)
    summary=dict(schema='r4_legal_responsibility_refinement_summary_v1',scene_id=native.scene_id,
        parent_candidate_hash=canonical_hash(native.candidate),new_candidate_hash=family['candidate_hash'],
        original_static_row_denominator=n,parent_U_row_denominator=int(parent_u.sum()),
        new_operable_row_count=int(selected.sum()),fixed_keep_parent_rows=int((parent_u&~selected).sum()),
        groups=len(groups),possible_routes=len(routes),group_counts=group_counts,
        parent_U_body_contribution=float(body_sum[parent_u].sum()),operable_body_contribution=float(body_sum[selected].sum()),
        parent_U_keep_contribution=float(keep_sum[parent_u].sum()),operable_keep_contribution=float(keep_sum[selected].sum()),
        body_contribution_coverage=float(body_sum[selected].sum()/body_sum[parent_u].sum()) if body_sum[parent_u].sum()>0 else None,
        legal_masks=receipts,new_frozen_scene_dir=str(scene),new_H0_domain=True,
        GT_used_for_selection=False,P_eligibility='uncalibrated_unknown',owner_truth_claimed=False,
        checkpoint_parameters_unchanged=native.unchanged(),forward_calls=3,adjoint_backward_calls=6,
        wall_seconds=time.monotonic()-start,human_verdict=None)
    write(args.output/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
