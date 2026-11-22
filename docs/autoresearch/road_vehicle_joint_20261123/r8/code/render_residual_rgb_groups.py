"""One original native sort/raster per view; additive five-group RGB attribution.

Default is CPU preparation. Only root schedules --execute. No optimizer or
parameter/asset mutation exists. Component isolation/subtraction is not used.
"""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from verify_joint_asset_cpu import read,write,pin,sha,SOURCE_SHA


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def png(path,value):
    from PIL import Image
    value=np.asarray(value)
    if value.ndim==3 and value.shape[0] in (1,3,4):value=value.transpose(1,2,0)
    if value.ndim==3 and value.shape[-1]==1:value=value[...,0]
    Image.fromarray(np.rint(np.clip(value,0,1)*255).astype(np.uint8)).save(path)
def bank(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit-contract',type=Path,required=True)
    p.add_argument('--asset',type=Path,required=True);p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',default=[86,110])
    p.add_argument('--selected-owner',type=int,default=21);p.add_argument('--execute',action='store_true');a=p.parse_args()
    assert a.frames==[86,110],'This registered readout has exactly two requested views'
    cfg=read(a.fit_contract);joint_path=a.asset/'joint_contract.json';joint=read(joint_path);proof_path=a.asset/'proof.json';proof=read(proof_path)
    lineage_path=a.asset/'colour_lineage.npz';lineage=bank(lineage_path);train_rows=lineage['road_world_rows']
    assert proof['source_field_sha256']==cfg['source']['sha256']==SOURCE_SHA
    assert proof['joint_contract_sha256']==sha(joint_path)
    assert sha(cfg['source']['field'])==SOURCE_SHA and a.selected_owner in [x['owner_id'] for x in joint['actors']]
    assert train_rows.ndim==1 and len(np.unique(train_rows))==len(train_rows)
    names=['trainable_road','fixed_old_road',f'retained_native_owner{a.selected_owner}','other_native_context','canonical_actor']
    sources={'fit_contract':pin(a.fit_contract),'asset_proof':pin(proof_path),'joint_contract':pin(joint_path),
       'colour_lineage':pin(lineage_path),'script':pin(__file__),'source_field_sha256':SOURCE_SHA}
    if not a.execute:
        print(json.dumps({'prepared_only':True,'frames':a.frames,'groups':names,'features':24,
          'feature_layout':'depth/alpha2 + five groups*(premult RGB3 + alpha1) + pad2',
          'expected_native_F_if_root_executes':2,'actual_nativeF':0,'nativeB':0,'updates':0,'diffusion_calls':0,'sources':sources}));return
    assert not a.output.exists(),'Preserve previous runs; use a new output directory'
    a.output.mkdir(parents=True);start=time.time();attempted=0;completed=0;rt=None;rows=[];error=None
    try:
        import torch
        from PIL import Image,ImageDraw
        from joint_identity_adapter import JointIdentityAdapter
        torch.set_num_threads(2);joint=dict(joint);joint['frames']=a.frames
        rt=JointIdentityAdapter(joint,a.output/'native')
        train=torch.as_tensor(train_rows,device='cuda',dtype=torch.long)
        with torch.no_grad():
            for raw in a.frames:
                folder=a.output/f'{raw:03d}';folder.mkdir()
                packet=rt.packet(raw);context,roadgroup,owners,gids,offset=rt.context[raw]
                n=len(packet['means3D']);nc=len(owners);assert nc==len(context['means3D'])
                group=torch.full((n,),-1,device='cuda',dtype=torch.long)
                roadpos=torch.nonzero(roadgroup==1).reshape(-1)
                local=gids[roadpos]-offset;assert (local>=0).all() and (local<len(rt.road_active)).all()
                world=torch.as_tensor(rt.road_active,device='cuda',dtype=torch.long)[local]
                is_train=torch.isin(world,train);group[roadpos]=torch.where(is_train,0,1)
                nativepos=torch.nonzero(roadgroup==0).reshape(-1)
                group[nativepos]=torch.where(owners[nativepos]==a.selected_owner,2,3)
                canonical_rows=sum(len(field.xyz) for _,field in rt.actors);assert nc+canonical_rows==n
                group[nc:]=4
                onehot=torch.nn.functional.one_hot(group,num_classes=5).to(packet['colors_precomp'].dtype)
                assert (group>=0).all() and torch.equal(onehot.sum(1),torch.ones(n,device='cuda'))
                features=torch.zeros((n,24),device='cuda',dtype=packet['colors_precomp'].dtype)
                features[:,0]=packet['features'][:,0];features[:,1]=1
                for index in range(5):
                    features[:,2+4*index:5+4*index]=packet['colors_precomp']*onehot[:,index,None]
                    features[:,5+4*index]=onehot[:,index]
                packet=dict(packet,features=features)
                attempted+=1
                native=rt.scene._bindings().raster(rt.cameras[raw],rt.scene.pipe,0,packet,rt.scene._background)
                completed+=1;rt.calls+=1
                env=rt.scene.environment(rt.cameras[raw].get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                rendered=native['rgb']+(1-native['alpha'])*env
                ar=lambda x:x.detach().cpu().numpy()
                native_rgb=ar(native['rgb']);rgb=ar(rendered);alpha=ar(native['alpha']).squeeze();extra=ar(native['extra'])
                assert extra.shape[0]>=20,extra.shape
                group_rgb=np.stack([extra[4*i:4*i+3] for i in range(5)])
                group_alpha=np.stack([extra[4*i+3] for i in range(5)])
                assert all(np.isfinite(x).all() for x in [native_rgb,rgb,alpha,group_rgb,group_alpha])
                ref_path=a.expected/f'{raw:03d}'/'joint_render.npz';ref=bank(ref_path)
                assert rgb.dtype==ref['rgb'].dtype and rgb.shape==ref['rgb'].shape and rgb.tobytes()==ref['rgb'].tobytes(),'Native RGB+env changed'
                assert alpha.dtype==ref['alpha'].dtype and alpha.shape==ref['alpha'].shape and alpha.tobytes()==ref['alpha'].tobytes(),'Native alpha changed'
                rgb_sum_error=float(abs(group_rgb.sum(0)-native_rgb).max());alpha_sum_error=float(abs(group_alpha.sum(0)-alpha).max())
                assert rgb_sum_error<=2e-5 and alpha_sum_error<=2e-5,(rgb_sum_error,alpha_sum_error)
                np.savez_compressed(folder/'rgb_groups.npz',group_RGB_premult=group_rgb,group_alphaT=group_alpha,
                  group_names=np.asarray(names),rgb_native_premult=native_rgb,rgb_with_environment=rgb,
                  native_alpha=alpha,environment_RGB=ar(env),camera_K=np.array([[rt.cameras[raw].fx,0,rt.cameras[raw].cx],
                    [0,rt.cameras[raw].fy,rt.cameras[raw].cy],[0,0,1]],np.float64),
                  selected_owner=np.asarray(a.selected_owner),group_row_counts=ar(onehot.sum(0)).astype(np.int64))
                png(folder/'shared_native.png',rgb);png(folder/'native_premult.png',native_rgb)
                for index,name in enumerate(names):png(folder/f'{index}_{name}_RGB_premult.png',group_rgb[index]);png(folder/f'{index}_{name}_alphaT.png',group_alpha[index])
                masks_path=Path(cfg['rectified_masks'])/f'{raw:03d}'/'rectified_masks.npz';masks=bank(masks_path);proxy=masks['actor_visible_proxy'].astype(bool)
                yy,xx=np.nonzero(proxy);assert len(xx)>0
                bbox=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
                roi=(max(bbox[0]-30,0),max(bbox[3]-12,0),min(bbox[2]+30,rgb.shape[2]),min(bbox[3]+100,rgb.shape[1]))
                x0,y0,x1,y1=roi;width=x1-x0;height=y1-y0
                images=[('actual shared RGB',rgb)]+list(zip(names,group_rgb))+[('native premult RGB',native_rgb)]
                panel=Image.new('RGB',(width*len(images),height+32),'#111827');draw=ImageDraw.Draw(panel)
                for index,(label,value) in enumerate(images):
                    crop=np.rint(np.clip(value[:,y0:y1,x0:x1].transpose(1,2,0),0,1)*255).astype(np.uint8)
                    panel.paste(Image.fromarray(crop),(index*width,32));draw.text((index*width+4,5),label,fill='white')
                panel.save(folder/'underbody_group_RGB_native_crop.png')
                roi_alpha=group_alpha[:,y0:y1,x0:x1];roi_rgb=group_rgb[:,:,y0:y1,x0:x1]
                row={'raw':raw,'expected_native_float':pin(ref_path),'masks':pin(masks_path),'all_rows_strict_partition':True,
                  'packet_rows':n,'context_rows':nc,'group_row_counts':ar(onehot.sum(0)).astype(int).tolist(),
                  'group_active_row_counts':[int(((group==i)&packet['mask']).sum()) for i in range(5)],
                  'native_RGB_with_env_byteexact_final':True,'native_alpha_byteexact_final':True,
                  'group_RGB_sum_max_abs_native_premult':rgb_sum_error,'group_alphaT_sum_max_abs_native_alpha':alpha_sum_error,
                  'rectified_actor_proxy_bbox_xyxy':bbox,'fixed_relative_lower_edge_ROI_xyxy':roi,
                  'ROI_selection':'proxy bbox x +/-30; lower edge -12 to +100; image clipped; chosen without RGB ranking',
                  'ROI_group_RGB_sum_channels':roi_rgb.sum((2,3),dtype=np.float64).tolist(),
                  'ROI_group_alphaT_sum':roi_alpha.sum((1,2),dtype=np.float64).tolist(),
                  'ROI_group_signed_red_excess_R_minus_meanGB':(roi_rgb[:,0]-.5*(roi_rgb[:,1]+roi_rgb[:,2])).sum((1,2),dtype=np.float64).tolist(),
                  'group_float':pin(folder/'rgb_groups.npz'),'native_crop':pin(folder/'underbody_group_RGB_native_crop.png'),
                  'actual_nativeF':1,'nativeB':0,'updates':0}
                save_json(folder/'receipt.json',row);rows.append(row);print(json.dumps(row),flush=True)
        rt.assert_fixed()
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
    finally:
        save_json(a.output/'receipt.json',{'schema':'R9.same_native_sorted_five_group_RGB_readout.v1','complete':error is None,'error':error,
          'sources':sources,'frames_requested':a.frames,'groups':names,'rows':rows,'native_F_attempts':attempted,
          'actual_nativeF':completed,'nativeB':0,'updates':0,'diffusion_calls':0,
          'single_same_sorted_native_raster_per_view':True,'component_difference_or_paste_used':False,
          'native_owner_group_is_not_automatically_A9':True,'model_geometry_source_or_loss_modified':False,'seconds':time.time()-start})


if __name__=='__main__':main()
