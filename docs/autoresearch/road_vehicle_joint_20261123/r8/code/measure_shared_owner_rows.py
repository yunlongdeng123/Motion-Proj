"""Explicit scheduled GPU measurement: one shared native F + one colors VJP per view.

No parameter update. Feature alphaT partitions and per-row mass identify actual
native static/dynamic owners/global IDs. Existing fixed V/proxy/K masks only.
"""
from pathlib import Path
import argparse,json,time,numpy as np
from PIL import Image
from joint_native_adapter import JointNativeAdapter,read,bank,sha

def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def npy(x):return x.detach().cpu().numpy()
def png(path,x):Image.fromarray(np.rint(np.clip(npy(x),0,1)*255).astype(np.uint8)).save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--contract',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,nargs='+',default=[86,110]);parser.add_argument('--measure',action='store_true');a=parser.parse_args()
    cfg=read(a.contract)
    if 'base_contract' in cfg:cfg=dict(read(cfg['base_contract']),**{k:v for k,v in cfg.items() if k!='base_contract'})
    cfg['frames']=a.frames
    if not a.measure:
        print(json.dumps(dict(prepared=True,actual_nativeF=0,actual_nativeB=0,expected_F=len(a.frames),expected_B=len(a.frames),requires_explicit_measure=True)));return
    import torch
    rt=JointNativeAdapter(cfg,a.output);started=time.time();rows=[];error=None
    try:
        proof=read(cfg['scene_proof']);ns=proof['topology']['static_rows'];nd=proof['topology']['dynamic_rows'];selected=set(x['owner_id'] for x in cfg['actors'])
        for raw in a.frames:
            folder=a.output/f'{raw:03d}';folder.mkdir()
            packet=rt.packet(raw);n=len(packet['means3D']);context,roadgroup,owner,gid,offset=rt.context[raw];nc=len(gid)
            branch=torch.full((n,),3,device='cuda',dtype=torch.long);owners=torch.full((n,),-1,device='cuda',dtype=torch.long);gids=torch.full((n,),-1,device='cuda',dtype=torch.long)
            branch[:nc]=torch.where(roadgroup==1,2,torch.where(gid<ns,0,1));owners[:nc]=owner;gids[:nc]=gid
            selected_native=torch.zeros(n,device='cuda',dtype=torch.bool)
            for value in selected:selected_native|=owners==value
            groups=[('road',branch==2),('selected_actor_field',branch==3),
                ('native_static_unowned',(branch==0)&(owners<=0)),('native_static_selected_owner',(branch==0)&selected_native),
                ('native_static_other_owner',(branch==0)&(owners>0)&~selected_native)]
            unique,count=torch.unique(owners[branch==1],return_counts=True)
            choices=sorted(zip(npy(unique).tolist(),npy(count).tolist()),key=lambda x:(-x[1],x[0]))
            keep_ids=[x[0] for x in choices[:16]]
            for value in sorted(keep_ids):groups.append((f'native_dynamic_owner{value}',(branch==1)&(owners==value)))
            remaining=(branch==1)&~torch.isin(owners,torch.tensor(keep_ids,device='cuda',dtype=owners.dtype))
            if remaining.any():groups.append(('native_dynamic_other_owners',remaining))
            assert len(groups)<=22
            feature=torch.zeros((n,24),device='cuda',dtype=packet['means3D'].dtype)
            cam=rt.cameras[raw];feature[:,0]=packet['means3D']@cam.world_view_transform[:3,2]+cam.world_view_transform[3,2];feature[:,1]=1
            partition=torch.zeros(n,device='cuda',dtype=torch.long)
            for i,(_,mask) in enumerate(groups):feature[:,2+i]=mask.float();partition+=mask.long()
            assert torch.all(partition==1)
            colors=packet['colors_precomp'].detach().clone().requires_grad_(True);packet=dict(packet,features=feature,colors_precomp=colors)
            out=rt.scene._bindings().raster(cam,rt.scene.pipe,0,packet,rt.scene._background);rt.calls+=1
            env=rt.scene.environment(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1);rgb=out['rgb']+(1-out['alpha'])*env
            prior=bank(Path(cfg['baseline_output'])/f'{raw:03d}/joint_render.npz');component=bank(Path(cfg['baseline_output'])/f'{raw:03d}/component_render.npz')
            masks=bank(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz')
            prior_actor=prior['actor_alphaT'];iso=component['actor_alpha'].squeeze()
            real=masks.get('actor_visible',np.zeros_like(iso,bool))&(iso>=.95)&(prior_actor<.95)
            proxy=masks['actor_visible_proxy']&(iso>=.95)&(prior_actor<.95);K=masks['protected'].astype(bool)
            rois=[real,proxy,K];names=['strictV_iso95_jointbelow95','proxy_iso95_jointbelow95','corrected_K']
            loss=sum((rgb[channel]*torch.as_tensor(mask,device='cuda')).sum() for channel,mask in enumerate(rois))
            gradient=torch.autograd.grad(loss,colors)[0];rt.backward_calls+=1
            rgb_error=float(abs(rgb-torch.as_tensor(prior['rgb'],device='cuda')).max())
            alpha_error=float(abs(out['alpha'].squeeze()-torch.as_tensor(prior['alpha'],device='cuda')).max())
            measured=[]
            for i,(name,mask) in enumerate(groups):
                contribution=out['extra'][i];png(folder/(name+'_alphaT.png'),contribution)
                measured.append(dict(group=name,packed_rows=int(mask.sum()),ROI_mass={r:float(gradient[mask,k].sum()) for k,r in enumerate(names)}))
            # Exact semantic owner breakdown comes from row VJP even if a rare full-image layer was aggregated.
            byowner=[]
            for b,label in [(0,'static'),(1,'dynamic'),(2,'road'),(3,'selected_actor')]:
                for value in torch.unique(owners[branch==b]).tolist():
                    take=(branch==b)&(owners==value)
                    byowner.append(dict(branch=label,owner=int(value),ROI_mass={r:float(gradient[take,k].sum()) for k,r in enumerate(names)}))
            target=(gradient[:,:2]>0).any(1);inv=torch.as_tensor(np.linalg.inv(rt.transform).astype(np.float32),device='cuda')
            world=packet['means3D'][target]@inv[:3,:3].T+inv[:3,3]
            np.savez_compressed(folder/'actual_contributing_rows.npz',packet_row=npy(torch.nonzero(target).reshape(-1)),native_global_id=npy(gids[target]),branch=npy(branch[target]),
                semantic_owner=npy(owners[target]),means_model=npy(packet['means3D'][target]),means_world=npy(world),scales_model=npy(packet['scales'][target]),
                opacity=npy(packet['opacities'][target]),ROI_mass=npy(gradient[target]),ROI_names=np.asarray(names))
            conservation=[dict(ROI=name,pixels=int(mask.sum()),row_mass=float(gradient[:,i].sum()),joint_alpha_mass=float((out['alpha'].squeeze()*torch.as_tensor(mask,device='cuda')).sum())) for i,(name,mask) in enumerate(zip(names,rois))]
            record=dict(raw=raw,actual_nativeF=1,actual_nativeB=1,parameter_updates=0,feature_groups=measured,exact_branch_owner_mass=byowner,regions=conservation,
                partition_alpha_max_abs=float(abs(out['extra'][:len(groups)].sum(0)-out['alpha'].squeeze()).max()),baseline_RGB_max_abs=rgb_error,baseline_alpha_max_abs=alpha_error,
                row_gradient_min=float(gradient.min()),actual_row_packet=str(folder/'actual_contributing_rows.npz'),row_packet_SHA=sha(folder/'actual_contributing_rows.npz'),
                any_K_touched_target_contributing_rows=int((target&(gradient[:,2]>0)).sum()),
                geometry_appearance_unchanged=True,low_owner_not_automatically_wrong_occlusion=True)
            write(folder/'receipt.json',record);rows.append(record);print(json.dumps(dict(frame_ready=raw,baseline_RGB_max_abs=rgb_error,regions=conservation,owners=byowner)),flush=True)
            del loss,gradient,out,colors,packet
        rt.assert_fixed()
    except BaseException as e:
        error=dict(type=type(e).__name__,message=str(e));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R8.shared_native_exact_owner_measurement.v1',complete=error is None,error=error,
            actual_nativeF=rt.calls,actual_nativeB=rt.backward_calls,optimizer_updates=0,new_teacher_calls=0,rows=rows,seconds=time.time()-started,
            no_gating_applied=True,R9_source_acceptance_pending=True,K_protection_requires_all_views_before_any_future_update=True))


if __name__=='__main__':main()
