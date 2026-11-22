"""Root-scheduled five-view row VJP and zero-step local retirement diagnostic.

No optimizer, new Gaussian, asset/default mutation, or whole-actor deletion.
Every packed context row is saved; absent-view candidates remain unknown.
"""
from pathlib import Path
import argparse,time
import numpy as np
from PIL import Image
from r10_common import contracts,read,load,write,sha,source_gate,npy,LocalGateAdapter,srgb_to_linear_numpy


def png(path,value):
    value=np.asarray(value)
    if value.ndim==3 and value.shape[0] in [1,3]:value=value.transpose(1,2,0)
    if value.ndim==3 and value.shape[-1]==1:value=value[...,0]
    Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).save(path)


def pin(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def evaluation(cfg):
    root=Path(cfg['fixed_eval']);receipt=read(root/'receipt.json');assert receipt['complete'] and receipt['optimization_enabled'] is False
    assert receipt['source_field_sha256']==cfg['source']['sha256']
    for row in receipt['rows']:assert sha(row['fixed_eval']['path'])==row['fixed_eval']['sha256']
    return root,receipt


def metrics(rgb,alpha,reference,ev):
    target=ev['target_common_linear'].transpose(2,0,1);linear=srgb_to_linear_numpy(rgb)*float(ev['exposure_gain']);m=ev['ground_eval'];w=ev['weight'][m]
    residual=linear[:,m]-target[:,m];loss=float(((residual**2).mean(0)*w).sum()/w.sum()) if w.sum()>0 else None
    result=dict(fixed_ground_pixels=int(m.sum()),fixed_ground_weighted_MSE=loss,domains={})
    for name in ['full_original_band','ground_band_before_support','ground_eval','body_protected','K_protected','red_diagnostic','blue_cyan_diagnostic','dark_context_diagnostic']:
        d=ev[name];change=abs(rgb-reference['rgb'])[:,d]
        result['domains'][name]=dict(pixels=int(d.sum()),scene_alpha_lt95=int((d&(alpha<.95)).sum()),scene_alpha_lt50=int((d&(alpha<.5)).sum()),
            RGB_delta_max=float(change.max()) if d.any() else None,RGB_delta_mean=float(change.mean()) if d.any() else None,
            RGB_mean=rgb[:,d].mean(1).tolist() if d.any() else None,unknown_target_pixels=int((d&~m).sum()))
    outside=~ev['full_original_band'];result['outside_full_band']=dict(RGB_delta_max=float(abs(rgb-reference['rgb'])[:,outside].max()),
        changed_pixels_above1e_6=int((outside&(abs(rgb-reference['rgb']).max(0)>1e-6)).sum()))
    return result


def save_baseline(path,rt,raw,out,rgb):
    alpha=npy(out['alpha']).squeeze();value=dict(rgb=npy(rgb),alpha=alpha,
        road_alphaT=npy(out['extra'][0]),actor_alphaT=npy(out['extra'][1]),context_alphaT=npy(out['extra'][2]),
        unified_actor_alphaT=npy(out['extra'][3:3+len(rt.actors)]),
        depth_expected_m=npy(out['depth_sum']).squeeze()/rt.meter_to_model/np.maximum(alpha,1e-8))
    np.savez_compressed(path,**value);png(path.with_name('shared.png'),value['rgb']);png(path.with_name('alpha.png'),alpha)
    return value


def measure(args,cfg,base,evroot,evreceipt):
    import torch
    args.output.mkdir(parents=True,exist_ok=False);rt=None;B=0;attempts=0;error=None;started=time.time();records=[];packets={}
    try:
        rt=LocalGateAdapter(base,args.output/'native');proof=read(base['scene_proof']);native_end=proof['topology']['static_rows']+proof['topology']['dynamic_rows']
        for raw in cfg['frames']:
            folder=args.output/f'{raw:03d}';folder.mkdir();ev=load(evroot/f'{raw:03d}/fixed_eval.npz');packet=rt.packet(raw)
            context,group,owner,gids,offset=rt.context[raw];nc=len(gids)
            columns=list(range(6,13));feature=packet['features'].detach().clone();feature[:,columns]=1;feature.requires_grad_(True)
            packet=dict(packet,features=feature)
            attempts+=1;v=rt.scene._bindings().raster(rt.cameras[raw],rt.scene.pipe,0,packet,rt.scene._background);rt.calls+=1
            env=rt.scene.environment(rt.cameras[raw].get_world_directions(False).permute(1,2,0)).permute(2,0,1);rgb=v['rgb']+(1-v['alpha'])*env
            scopes=[ev['K_protected'],ev['body_protected'],ev['ground_band_before_support'],ev['ground_eval'],ev['red_diagnostic'],ev['blue_cyan_diagnostic'],ev['dark_context_diagnostic']]
            loss=sum((v['extra'][column-2]*torch.as_tensor(mask,device='cuda')).sum() for column,mask in zip(columns,scopes))
            grad=torch.autograd.grad(loss,feature)[0][:,columns];B+=1;assert float(grad.min())>=0
            conservation=[];aa=npy(v['alpha']).squeeze().astype(np.float64);gg=npy(grad).astype(np.float64)
            for index,(name,mask,column) in enumerate(zip(cfg['VJP_channels'],scopes,columns)):
                direct=float(aa[mask].sum());row_mass=float(gg[:,index].sum());one_error=float(abs(npy(v['extra'][column-2])-aa).max())
                assert one_error<3e-6 and abs(row_mass-direct)<max(.01,2e-6*mask.sum()),(raw,name,row_mass,direct,one_error)
                conservation.append(dict(region=name,pixels=int(mask.sum()),row_mass=row_mass,direct_alpha_mass=direct,abs_error=abs(row_mass-direct),constant_one_alpha_max_abs=one_error))
            rows=dict(global_row_ids=npy(gids),owner_ids=npy(owner).astype(np.int32),native_context=npy(group==0),ROI_alphaT_mass=npy(grad[:nc]),
                ROI_names=np.asarray(cfg['VJP_channels']))
            assert np.all(rows['global_row_ids'][rows['native_context']]<native_end)
            np.savez_compressed(folder/'all_context_rows.npz',**rows);packets[raw]=rows
            baseline=save_baseline(folder/'baseline_joint.npz',rt,raw,v,rgb);record=metrics(baseline['rgb'],baseline['alpha'],baseline,ev)
            record.update(raw=raw,all_packed_context_and_road_rows=nc,all_native_context_rows=int(rows['native_context'].sum()),
                all_row_alphaT_saved=True,K_pixels=int(scopes[0].sum()),body_pixels=int(scopes[1].sum()),ground_pixels=int(scopes[2].sum()),
                packet=pin(folder/'all_context_rows.npz'),baseline=pin(folder/'baseline_joint.npz'),feature_VJP_conservation=conservation)
            write(folder/'metrics.json',record);records.append(record);print(dict(frame=raw,context_rows=nc,nativeF=1,nativeB=1),flush=True)
            del feature,packet,v,rgb,grad,loss
        # Only present in ALL full packed-row packets is accepted as measured.
        union=[]
        for raw,z in packets.items():
            allowed=z['owner_ids']<=0
            for value in cfg['deleted_target_owner_ids']:allowed|=z['owner_ids']==value
            take=z['native_context']&allowed&(z['ROI_alphaT_mass'][:,2]>0)
            union.append(z['global_row_ids'][take])
        candidates=np.unique(np.concatenate(union));mass=np.zeros((len(candidates),len(cfg['frames']),7),np.float32);present=np.zeros(mass.shape[:2],bool);owners=np.full(mass.shape[:2],-999,np.int32)
        for column,raw in enumerate(cfg['frames']):
            z=packets[raw];ids=z['global_row_ids'];assert (np.diff(ids)>0).all();where=np.searchsorted(ids,candidates);ok=where<len(ids);ok[ok]&=ids[where[ok]]==candidates[ok]
            present[:,column]=ok;mass[ok,column]=z['ROI_alphaT_mass'][where[ok]];owners[ok,column]=z['owner_ids'][where[ok]]
        permanent=npy(rt.preserve);keep=present.all(1)&(mass[:,:,0]==0).all(1)&(mass[:,:,1]==0).all(1)&~np.isin(candidates,permanent)
        allowed=(owners<=0)
        for value in cfg['deleted_target_owner_ids']:allowed|=owners==value
        keep&=allowed.all(1);ids=candidates[keep]
        np.savez_compressed(args.output/'native_candidates.npz',global_row_ids=ids,frames=np.asarray(cfg['frames']),
            allview_K_mass=mass[keep,:,0],allview_body_mass=mass[keep,:,1],allview_ground_mass=mass[keep,:,2],allview_certified_ground_mass=mass[keep,:,3],
            allview_red_mass=mass[keep,:,4],allview_blue_mass=mass[keep,:,5],allview_dark_mass=mass[keep,:,6],owners_by_view=owners[keep])
        np.savez_compressed(args.output/'native_gate0.npz',global_row_ids=ids,gate=np.zeros(len(ids),np.float32))
        write(args.output/'candidate_selection.json',dict(initial_ground_contributing_allowed_union=len(candidates),safe_rows=len(ids),
            missing_view_rows=int((~present.all(1)).sum()),Ktouch_rows=int((mass[:,:,0]>0).any(1).sum()),bodytouch_rows=int((mass[:,:,1]>0).any(1).sum()),
            old_permanent_guard_rows=int(np.isin(candidates,permanent).sum()),no_missing_row_was_assumed_Kzero=True,
            native_selected_owner_and_other_retained_vehicles_excluded=True,old_guard_preserved=True))
        rt.assert_fixed();write(args.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(args.output/'receipt.json',dict(schema='R10.all_context_local_alphaT_VJP.v1',complete=error is None,error=error,
            native_F_attempts=attempts,actual_nativeF=rt.calls if rt else 0,nativeB=B,updates=0,new_diffusion_calls=0,
            seconds=time.time()-started,rows=records,source_field_sha256=cfg['source']['sha256'],baseline_contract_sha256=sha(cfg['baseline_contract']),
            fixed_eval_receipt_sha256=sha(evroot/'receipt.json'),actor_original_r3_frozen=True,road400_frozen=True,all_centers_scales_tracks_fixed=True))
    sidecar=dict(schema='R10.native_global_opacity_gate.v1',complete=True,decision='PASS',candidate_rows=int(len(ids)),
        baseline_contract_sha256=sha(cfg['baseline_contract']),candidate_npz_sha256=sha(args.output/'native_candidates.npz'),gate_npz_sha256=sha(args.output/'native_gate0.npz'),
        actual_attribution_receipt_sha256=sha(args.output/'receipt.json'),fixed_eval_receipt_sha256=sha(evroot/'receipt.json'),
        all_five_K_body_exact_zero=True,old_permanent_guard_preserved=True,old_context_plan_sha256=sha(base['context_plan']),native_source_pins=proof['native_sources'],
        native_topology=proof['topology'],gate_only_decrease=True,native21_and_other_retained_actors_frozen=True,new_geometry=False,
        candidate_positive_region='full_noncolor_ground_band including unknown-target pixels',training_target_region='certified_ground_only; unknown never RGB ground truth',
        eligible_deleted_target_owner_ids=cfg['deleted_target_owner_ids'],all_other_positive_owner_ids_frozen=True,
        zero_step_quality_not_yet_accepted=True,scope_PASS_does_not_mean_quality_PASS=True,optimization_enabled=False)
    write(args.output/'native_gate0.proof.json',sidecar);print(dict(complete=True,candidate_rows=len(ids),receipt=pin(args.output/'receipt.json'),proof=pin(args.output/'native_gate0.proof.json')))


def retire0(args,cfg,base,evroot):
    selection=read(args.measurement/'native_gate0.proof.json');gate=load(args.measurement/'native_gate0.npz')
    assert selection['complete'] and selection['decision']=='PASS' and selection['baseline_contract_sha256']==sha(cfg['baseline_contract'])
    assert selection['gate_npz_sha256']==sha(args.measurement/'native_gate0.npz') and (gate['gate']==0).all()
    if not len(gate['global_row_ids']):print(dict(not_run=True,reason='No safe local rows; switch to separately qualified local representation',nativeF=0));return
    import torch
    args.output.mkdir(parents=True,exist_ok=False);rt=None;error=None;rows=[]
    try:
        rt=LocalGateAdapter(base,args.output/'native',gate['global_row_ids'],cfg['deleted_target_owner_ids'])
        with torch.no_grad():
            for raw in cfg['frames']:
                ev=load(evroot/f'{raw:03d}/fixed_eval.npz');ref=load(args.measurement/f'{raw:03d}/baseline_joint.npz');out=rt.render(raw)
                rgb=npy(out['rgb']);alpha=npy(out['alpha']).squeeze();preserve=ev['K_protected']|ev['body_protected']
                assert rgb[:,preserve].tobytes()==ref['rgb'][:,preserve].tobytes() and alpha[preserve].tobytes()==ref['alpha'][preserve].tobytes(),'Protected body/K changed'
                folder=args.output/f'{raw:03d}';folder.mkdir();np.savez_compressed(folder/'joint_render.npz',rgb=rgb,alpha=alpha,
                    road_alphaT=npy(out['road_alpha']),actor_alphaT=npy(out['actor_alpha']),context_alphaT=npy(out['context_alpha']),unified_actor_alphaT=npy(out['unified_actor_alpha']))
                png(folder/'shared.png',rgb);png(folder/'alpha.png',alpha)
                record=metrics(rgb,alpha,ref,ev);record.update(raw=raw,accepted_body_and_K_RGBA_exact=True)
                write(folder/'metrics.json',record);rows.append(record)
        rt.assert_fixed()
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(args.output/'receipt.json',dict(schema='R10.zero_step_local_native_retirement_diagnostic.v1',complete=error is None,error=error,
            nativeF=rt.calls if rt else 0,nativeB=0,updates=0,rows=rows,candidate_rows=len(gate['global_row_ids']),
            native_gate_proof=pin(args.measurement/'native_gate0.proof.json'),zero_step_only=True,new_local_geometry=False,
            official_training_not_started=True,formal_updates_so_far=400,formal_budget_ceiling=2000,next_phase_max600_pending_root_review=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['measure','retire0'],default='measure');p.add_argument('--measurement',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg,base=contracts(a.contract);gate=source_gate(cfg,strict=a.execute);evroot,evreceipt=evaluation(cfg)
    assert cfg['deleted_target_owner_ids']==[10] and all(s['owner_id'] not in cfg['deleted_target_owner_ids'] for s in base['actors'])
    if not a.execute:print(dict(prepared=True,mode=a.mode,source_gate=gate,eval_receipt_sha256=sha(evroot/'receipt.json'),expected_F=5,expected_B=5 if a.mode=='measure' else 0,updates=0));return
    if a.mode=='measure':measure(a,cfg,base,evroot,evreceipt)
    else:assert a.measurement;retire0(a,cfg,base,evroot)


if __name__=='__main__':main()
