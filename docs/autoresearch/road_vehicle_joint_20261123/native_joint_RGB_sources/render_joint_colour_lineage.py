"""Actual depth-sorted projection of supervision labels, never raw-source RGB.

The fitted shader is unchanged. Continuous sampled shared-field weights label
rows; they are not causal contributions or a mixing equation for raw photos.
Root schedules this optional five-F, zero-B readout after formal/fresh checks.
"""
from pathlib import Path
import argparse,time
import numpy as np
from joint_rgb_common import configuration,source_gate,read,load,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--asset',type=Path,required=True);p.add_argument('--expected',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    cfg,_=configuration(a.contract);gate=source_gate(cfg,strict=a.execute)
    if not a.execute:print(dict(prepared=True,nativeF=0,nativeB=0,updates=0,expected_F=len(cfg['frames']),gate=gate));return
    import torch
    from joint_identity_adapter import JointIdentityAdapter
    from fit_joint_rgb import png
    proof=read(a.asset/'proof.json');assert proof['source_field_sha256']==cfg['source']['sha256']
    lineage=load(a.asset/'colour_lineage.npz');ids=lineage['road_world_rows'];weights=lineage['road_source_weight']
    assert weights.shape==(len(ids),3) and np.isfinite(weights).all() and (weights>=0).all() and abs(weights.sum(1)-1).max()<2e-6
    rt=JointIdentityAdapter(read(a.asset/'joint_contract.json'),a.output/'native');a.output.mkdir(parents=True,exist_ok=True)
    names=['retained_native_context','unchanged_road','same_world_supervision','similar_photo_supervision','continuous_teacher_influence','selected_canonical_actor']
    palette=np.array([[.30,.35,.41],[.45,.49,.54],[.22,.80,.56],[.91,.65,.27],[.66,.43,.88],[.25,.71,.85]],np.float32)
    started=time.time();records=[];error=None
    try:
        with torch.no_grad():
            for raw in cfg['frames']:
                packet=rt.packet(raw);n=len(packet['means3D']);context,group,owners,gids,offset=rt.context[raw];nc=len(gids)
                label=torch.zeros((n,6),device='cuda');native=torch.nonzero(group==0).reshape(-1);road=torch.nonzero(group==1).reshape(-1)
                label[native,0]=1;label[road,1]=1;label[nc:,5]=1
                world=rt.road_active[(gids[road]-offset).cpu().numpy()]
                pos=np.searchsorted(ids,world);inside=pos<len(ids);inside[inside]&=ids[pos[inside]]==world[inside]
                selected=road[torch.as_tensor(inside,device='cuda')]
                label[selected,1]=0;label[selected,2:5]=torch.as_tensor(weights[pos[inside]],device='cuda')
                assert float(abs(label.sum(1)-1).max())<2e-6
                feature=packet['features'].clone();feature[:,2:8]=label
                feature[:,8]=(label[:,1:5].sum(1));feature[:,9]=label[:,5];feature[:,10]=label[:,0]
                # Keep a separate same-instance alias after source-label channels.
                for index,(spec,field) in enumerate(rt.actors):
                    assert 11+index<24;feature[:,11+index]=0
                    feature[:nc,11+index]=((group==0)&(owners==spec['owner_id'])).float()
                cursor=nc
                for index,(_,field) in enumerate(rt.actors):feature[cursor:cursor+len(field.xyz),11+index]=1;cursor+=len(field.xyz)
                slot=11+len(rt.actors);assert slot+3+len(rt.actors)<=24
                feature[:,slot:slot+3+len(rt.actors)]=0
                feature[selected,slot]=1;feature[road,slot+1]=1;feature[selected,slot+1]=0
                retired=rt.cfg.get('exclude_native_owner_ids',[])
                if retired:
                    feature[:nc,slot+2]=((group==0)&torch.isin(owners,torch.as_tensor(retired,device='cuda'))).float()
                for index,(spec,field) in enumerate(rt.actors):
                    feature[:nc,slot+3+index]=((group==0)&(owners==spec['owner_id'])).float()
                v=rt.scene._bindings().raster(rt.cameras[raw],rt.scene.pipe,0,dict(packet,features=feature),rt.scene._background);rt.calls+=1
                env=rt.scene.environment(rt.cameras[raw].get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                rgb=v['rgb']+(1-v['alpha'])*env;source=v['extra'][:6];ref=load(a.expected/f'{raw:03d}/joint_render.npz')
                rgbmax=float(abs(rgb-torch.as_tensor(ref['rgb'],device='cuda')).max());amax=float(abs(v['alpha'].squeeze()-torch.as_tensor(ref['alpha'],device='cuda')).max())
                assert rgbmax==0 and amax==0
                alpha_error=float(abs(source.sum(0)-v['alpha'].squeeze()).max())
                road_error=float(abs(source[1:5].sum(0)-torch.as_tensor(ref['road_alphaT'],device='cuda')).max())
                actor_error=float(abs(source[5]-torch.as_tensor(ref['actor_alphaT'],device='cuda')).max())
                context_error=float(abs(source[0]-torch.as_tensor(ref['context_alphaT'],device='cuda')).max())
                assert max(alpha_error,road_error,actor_error,context_error)<3e-6
                trainalpha=v['extra'][slot-2];fixedalpha=v['extra'][slot-1]
                split_error=float(abs(trainalpha+fixedalpha-torch.as_tensor(ref['road_alphaT'],device='cuda')).max());assert split_error<3e-6
                folder=a.output/f'{raw:03d}';folder.mkdir()
                display=torch.einsum('khw,kc->chw',source,torch.as_tensor(palette,device='cuda'))
                png(folder/'supervision_layers.png',display);png(folder/'actual_shared.png',rgb)
                for index,name in enumerate(names):png(folder/(name+'_alphaT.png'),source[index])
                for name,value in [('trainable_road',trainalpha),('fixed_old_road',fixedalpha),
                    ('retained_excluded_native_owner',v['extra'][slot]),('native_selected_instance',v['extra'][slot+1:slot+1+len(rt.actors)].sum(0))]:
                    png(folder/(name+'_alphaT.png'),value)
                np.savez_compressed(folder/'layer_integrals.npz',source_names=np.asarray(names),source_alphaT=source.cpu().numpy(),
                    actual_shared_RGB=rgb.cpu().numpy(),actual_scene_alpha=v['alpha'].squeeze().cpu().numpy(),
                    native_road_alphaT=v['extra'][6].cpu().numpy(),native_actor_alphaT=v['extra'][7].cpu().numpy(),
                    native_context_alphaT=v['extra'][8].cpu().numpy(),unified_actor_alphaT=v['extra'][9:9+len(rt.actors)].cpu().numpy(),
                    trainable_road_alphaT=trainalpha.cpu().numpy(),fixed_old_road_alphaT=fixedalpha.cpu().numpy(),
                    retained_excluded_native_owner_alphaT=v['extra'][slot].cpu().numpy(),
                    native_selected_instance_alphaT=v['extra'][slot+1:slot+1+len(rt.actors)].cpu().numpy())
                records.append(dict(raw=raw,RGB_max_abs_vs_final=rgbmax,alpha_max_abs_vs_final=amax,source_sum_scene_alpha_max_abs=alpha_error,
                    source_road_sum_native_road_alphaT_max_abs=road_error,actor_layer_max_abs=actor_error,context_layer_max_abs=context_error,
                    trainable_plus_fixed_road_alphaT_max_abs=split_error))
        rt.assert_fixed()
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R8.actual_joint_supervision_influence_readout.v1',complete=error is None,error=error,
            nativeF=rt.calls,nativeB=0,updates=0,seconds=time.time()-started,source_field_sha256=cfg['source']['sha256'],
            asset_proof_sha256=sha(a.asset/'proof.json'),lineage_sha256=sha(a.asset/'colour_lineage.npz'),source_names=names,palette=palette.tolist(),rows=records,
            row_weight_interpretation='sampled shared-field supervision provenance and continuous conditioning influence; not causal raw-source RGB contributions',
            fitted_RGB_is_not_raw_photo_or_teacher_RGB_average=True,unchanged_road_and_native_context_separate=True,
            excluded_native_owner_ids=rt.cfg.get('exclude_native_owner_ids',[]),selected_owner_ids=[s['owner_id'] for s,_ in rt.actors],
            identity_diagnostics_use_actual_loaded_tags_only=True,untagged_residuals_not_automatically_assigned_to_retired_actors=True,
            selected_actor_preserves_historical_A20_field_plus_real_supervised_DC=True,output_RGBA_unmodified=True if error is None else None))


if __name__=='__main__':main()
