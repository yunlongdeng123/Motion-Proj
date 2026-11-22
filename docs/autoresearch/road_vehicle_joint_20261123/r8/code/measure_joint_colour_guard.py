"""Scheduled actual colour-VJP guard: five shared F/B, no optimizer/update.

K uses all five corrected pinhole masks. Strict actor V is read only for the
two declared training views. Native context and old K-positive guards stay fixed.
"""
from pathlib import Path
import argparse,time
import numpy as np
from joint_rgb_common import configuration,source_gate,load,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--measure',action='store_true');a=p.parse_args()
    cfg,base=configuration(a.contract);gate=source_gate(cfg,strict=a.measure)
    if not a.measure:
        print(dict(prepared=True,newF=0,newB=0,updates=0,expected_actualF=len(cfg['frames']),expected_actualB=len(cfg['frames']),source_gate=gate));return
    # Import and construction only after both source gates pass.
    import torch
    from joint_native_adapter import JointNativeAdapter
    rt=JointNativeAdapter(base,a.output/'native');a.output.mkdir(parents=True,exist_ok=True)
    started=time.time();B=0;error=None;rows=[];road_mass=[];actor_mass={s['name']:[] for s,_ in rt.actors};actor_V={s['name']:[] for s,_ in rt.actors}
    try:
        for raw in cfg['frames']:
            packet=rt.packet(raw);n=len(packet['means3D']);context,roadgroup,owners,gids,offset=rt.context[raw];nc=len(gids)
            colors=packet['colors_precomp'].detach().clone().requires_grad_(True);packet=dict(packet,colors_precomp=colors)
            out=rt.scene._bindings().raster(rt.cameras[raw],rt.scene.pipe,0,packet,rt.scene._background);rt.calls+=1
            env=rt.scene.environment(rt.cameras[raw].get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            rgb=out['rgb']+(1-out['alpha'])*env
            masks=load(Path(cfg['rectified_masks'])/f'{raw:03d}/rectified_masks.npz');K=masks['protected'].astype(bool)
            V=masks['actor_visible'].astype(bool) if raw in cfg['train_frames'] else np.zeros_like(K)
            prior=load(Path(cfg['baseline_render'])/f'{raw:03d}/joint_render.npz')
            assert K.shape==tuple(rgb.shape[1:])
            err=float(abs(rgb-torch.as_tensor(prior['rgb'],device='cuda')).max())
            ae=float(abs(out['alpha'].squeeze()-torch.as_tensor(prior['alpha'],device='cuda')).max())
            assert err==0 and ae==0,'Zero-update source RGBA differs from R8 baseline'
            loss=(rgb[0]*torch.as_tensor(K,device='cuda')).sum()+(rgb[1]*torch.as_tensor(V,device='cuda')).sum()
            grad=torch.autograd.grad(loss,colors)[0];B+=1;assert float(grad.min())>=0
            roadpos=torch.nonzero(roadgroup==1).reshape(-1);local=gids[roadpos]-offset
            roadK=np.zeros(len(rt.road_active),np.float32);roadK[local.cpu().numpy()]=grad[roadpos,0].detach().cpu().numpy();road_mass.append(roadK)
            # Every canonical row is saved, including exact-zero and K-only rows.
            cursor=nc
            for spec,field in rt.actors:
                end=cursor+len(field.xyz);actor_mass[spec['name']].append(grad[cursor:end,0].detach().cpu().numpy())
                actor_V[spec['name']].append(grad[cursor:end,1].detach().cpu().numpy());cursor=end
            assert cursor==n
            record=dict(raw=raw,K_pixels=int(K.sum()),strict_train_V_pixels=int(V.sum()),
                source_RGB_max_abs=err,source_alpha_max_abs=ae,road_Kpositive_rows=int((roadK>0).sum()),
                canonical_Kpositive_rows={s['name']:int((actor_mass[s['name']][-1]>0).sum()) for s,_ in rt.actors},
                nativeF=1,nativeB=1,updates=0)
            rows.append(record);print(record,flush=True);del loss,grad,out,rgb,packet,colors
        mass=np.stack(road_mass);payload=dict(frames=np.asarray(cfg['frames']),road_active_world_rows=rt.road_active,
            road_K_mass=mass,road_Kpositive_union=(mass>0).any(0),road_Kzero_allviews=(mass==0).all(0))
        for spec,_ in rt.actors:
            name=spec['name'];m=np.stack(actor_mass[name]);v=np.stack(actor_V[name])
            payload[name+'_K_mass']=m;payload[name+'_Kpositive_union']=(m>0).any(0)
            payload[name+'_strict_train_V_mass']=v;payload[name+'_strict_train_Vpositive_union']=(v>0).any(0)
        np.savez_compressed(a.output/'colour_guard.npz',**payload);rt.assert_fixed()
        write(a.output/'camera_timestamp_audit.json',dict(rows=rt.camera_audit))
        write(a.output/'accepted_source_gate.json',gate)
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        write(a.output/'receipt.json',dict(schema='R8.all_five_view_actual_colour_K_guard.v1',complete=error is None,error=error,
            nativeF=rt.calls,nativeB=B,updates=0,new_diffusion_calls=0,seconds=time.time()-started,
            source_sha256=cfg['source']['sha256'],baseline_contract_sha256=sha(cfg['baseline_contract']),
            guard_sha256=sha(a.output/'colour_guard.npz') if (a.output/'colour_guard.npz').exists() else None,
            frames=cfg['frames'],rows=rows,all_road_and_canonical_rows_measured=True,
            native_context_and_old_K_guard_fixed=True,source_RGBA_exact=True if error is None else None,
            only_actual_positive_mass_freezes_rows=True,missing_K_rows_filled_zero=False))


if __name__=='__main__':main()
