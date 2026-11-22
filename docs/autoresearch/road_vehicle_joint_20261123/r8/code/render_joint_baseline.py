"""Explicit zero-update GPU baseline; no train/optimizer path."""
from pathlib import Path
import argparse,json,time,numpy as np
from PIL import Image
from joint_native_adapter import JointNativeAdapter,read,bank,sha

_EXECUTION={}


def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def array(x):return x.detach().cpu().numpy()
def png(path,x):
    x=array(x) if hasattr(x,'detach') else np.asarray(x)
    if x.ndim==3 and x.shape[0] in [1,3,4]:x=x.transpose(1,2,0)
    x=np.squeeze(x) if x.ndim==3 and x.shape[-1]==1 else x
    Image.fromarray(np.rint(np.clip(x,0,1)*255).astype(np.uint8)).save(path)
def compare(a,b):
    a=array(a).squeeze();b=np.asarray(b).squeeze()
    return dict(max_abs=float(abs(a-b).max()),MAE=float(abs(a-b).mean()),RGB8_max=int(abs(np.rint(np.clip(a,0,1)*255)-np.rint(np.clip(b,0,1)*255)).max()))


def _main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+');p.add_argument('--render',action='store_true');p.add_argument('--components',action='store_true');p.add_argument('--equivalence',action='store_true')
    a=p.parse_args();cfg=read(a.contract)
    if a.frames:cfg['frames']=a.frames
    if not a.render:
        actors=[]
        for s in cfg['actors']:
            assert sha(s['asset'])==s['sha256'];z=bank(s['asset']);actors.append(dict(name=s['name'],rows=len(z['xyz']),active=int(z['active'].sum()),sha256=s['sha256']))
        print(json.dumps(dict(CPU_only=True,frames=cfg['frames'],actors=actors,render_requires_explicit_flag=True,updates=0)));return
    import torch
    started=time.time();_EXECUTION.update(output=a.output,output_was_new=not a.output.exists(),started=started,rt=None,rows=[],frames=cfg['frames'])
    rt=JointNativeAdapter(cfg,a.output);_EXECUTION['rt']=rt
    save_json(a.output/'source_receipt.json',rt.source_receipt);save_json(a.output/'camera_timestamp_audit.json',rt.camera_audit)
    rows=_EXECUTION['rows']
    with torch.no_grad():
        for raw in cfg['frames']:
            folder=a.output/f'{raw:03d}';folder.mkdir();joint=rt.render(raw)
            for key in ['rgb','alpha','road_alpha','actor_alpha','context_alpha']:png(folder/(key+'.png'),joint[key])
            alpha=joint['alpha'].squeeze();depth=joint['depth_sum_m'].squeeze()/alpha.clamp_min(1e-8)
            owner=torch.stack([joint['road_alpha'],joint['actor_alpha'],joint['context_alpha']]).argmax(0)
            palette=torch.tensor([[.1,.8,.8],[1.,.55,.12],[.55,.55,.65]],device='cuda');rgb_owner=palette[owner].permute(2,0,1)*(alpha>.01)
            png(folder/'depth_owner_alphaT.png',rgb_owner);png(folder/'expected_depth_m_0_60.png',depth/60)
            np.savez_compressed(folder/'joint_render.npz',rgb=array(joint['rgb']),alpha=array(alpha),depth_expected_m=array(depth),
                road_alphaT=array(joint['road_alpha']),actor_alphaT=array(joint['actor_alpha']),context_alphaT=array(joint['context_alpha']))
            rec=dict(raw=raw,shared_raster_calls=1,owner_alpha_sum_max_abs=float(abs(joint['road_alpha']+joint['actor_alpha']+joint['context_alpha']-alpha).max()),
                finite=all(torch.isfinite(x).all().item() for x in joint.values()),scope='maps only; no old Brown/NN masks used as new metric denominators')
            if a.components or a.equivalence:
                road=rt.render(raw,'road');actor=rt.render(raw,'actor')
                for label,value in [('road_only',road),('actor_only',actor)]:
                    png(folder/(label+'_rgb.png'),value['rgb']);png(folder/(label+'_alpha.png'),value['alpha'])
                np.savez_compressed(folder/'component_render.npz',road_rgb=array(road['rgb']),road_alpha=array(road['alpha']).squeeze(),
                    actor_rgb=array(actor['rgb']),actor_alpha=array(actor['alpha']).squeeze(),
                    road_depth_sum_m=array(road['depth_sum_m']).squeeze(),actor_depth_sum_m=array(actor['depth_sum_m']).squeeze(),
                    road_depth_expected_m=array(road['depth_sum_m'].squeeze()/road['alpha'].squeeze().clamp_min(1e-8)),
                    actor_depth_expected_m=array(actor['depth_sum_m'].squeeze()/actor['alpha'].squeeze().clamp_min(1e-8)))
                rec['component_alpha_scope']='scene total alpha, road-only alpha and actor-only alpha are separate; low joint component ownership can be occlusion'
                overlap=(road['alpha'].squeeze()>.05)&(actor['alpha'].squeeze()>.05)
                rec['road_actor_isolated_overlap_pixels_unscoped']=int(overlap.sum())
                rec['actor_joint_vs_isolated_alpha_MAE_unscoped']=float(abs(joint['actor_alpha']-actor['alpha'].squeeze()).mean())
                if a.equivalence:
                    old=bank(Path(cfg['road_reference_root'])/f'{raw:03d}/render.npz')
                    rec['road_vs_R7_parent']={k:compare(road[k],old['road_'+k]) for k in ['rgb','alpha']}
                    if raw in cfg.get('legacy_reference_frames',[]):
                        oldactor=rt.legacy_reference(raw,True);oldjoint=rt.legacy_reference(raw,False)
                        rec['actor_vs_selected_r3']={k:compare(actor[k],array(oldactor[k])) for k in ['rgb','alpha']}
                        rec['shared_vs_selected_r3']={k:compare(joint[k],array(oldjoint[k])) for k in ['rgb','alpha']}
            rows.append(rec);save_json(folder/'receipt.json',rec);print(json.dumps(dict(frame_ready=True,**rec)),flush=True)
    rt.assert_fixed()
    receipt=dict(schema='R8.zero_update.shared_native_baseline.v1',complete=True,frames=cfg['frames'],actual_nativeF=rt.calls,actual_nativeB=0,
        optimizer_updates=0,new_diffusion_calls=0,Gaussian_insertion=0,road_and_actor_centers_fixed=True,normal_frames_fixed=True,rigid_tracks_fixed=True,
        one_actual_shared_depth_sorted_raster_per_joint_view=True,not_a_2D_layer_paste=True,not_claimed_original_DGGT=True,
        masks_formal_metrics='pending independently Brown-rectified masks; no legacy NN mask denominator claimed',
        rows=rows,trace=rt.trace,seconds=time.time()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated())
    save_json(a.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


def main():
    error=None;success=False
    try:
        _main();success=True
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        if _EXECUTION and _EXECUTION.get('output_was_new'):
            output=_EXECUTION['output'];output.mkdir(parents=True,exist_ok=True)
            rt=_EXECUTION.get('rt')
            execution=dict(schema='R8.zero_update.baseline_execution_accounting.v1',complete=success,error=error,
                frames_requested=_EXECUTION.get('frames'),completed_frame_rows=_EXECUTION.get('rows',[]),
                actual_nativeF=0 if rt is None else rt.calls,actual_nativeB=0 if rt is None else rt.backward_calls,
                optimizer_updates=0,new_diffusion_calls=0,trace=[] if rt is None else rt.trace,
                constructor_failed_before_any_raster=bool(rt is None and error),
                seconds=time.time()-_EXECUTION['started'],renderer_math_changed=False)
            save_json(output/'execution_receipt.json',execution)
            if not success:save_json(output/'failure_receipt.json',execution)


if __name__=='__main__':main()
