"""限定屏幕域的实际Gaussian候选删除；白底只填透射，不把不透明残留涂白。"""
import sys, json, time, argparse, gc
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/unknown_detection_v3/code')
from probe_detection import load_native,kwargs,attribution,authorized,write,save,torch,cpu,exclusive

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    authorized();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz)
        for i,f in enumerate(frames):
            raw=f['raw']
            if raw not in [98,113,114]:continue
            dest=a.output/f'{raw:03d}';dest.mkdir();src=a.source/f'{raw:03d}'
            inp=np.load(a.inputs/f'{raw:03d}/diagnostic_gate.npz');allowed=inp['allowed']
            state=np.load(src/'existing/maps.npz');body=np.load(src/'domains.npz')['maps'][2]>0
            # 输入来自用户既定局部查询，原生饱和查询仅用于诊断，绝不当训练标签。
            clipped=((state['native_rgb']<0).any(0)|(state['native_rgb']>1).any(0))&allowed
            query=(body|inp['user_contact_query']|clipped)&allowed
            maps=np.stack([query,allowed,~allowed,np.ones_like(allowed)]).astype('float32')
            cam,st,dy=kwargs(n,actor,f,i)
            with torch.no_grad():
                h=torch.cat((st['means3D'],torch.ones_like(st['means3D'][:,:1])),1)
                ndc=h@cam.full_proj_transform;ndc=ndc[:,:3]/ndc[:,3:4]
                screen=cpu(torch.stack(((ndc[:,0]+1)*cam.image_width/2-.5,(ndc[:,1]+1)*cam.image_height/2-.5),1))
                z=cpu((h@cam.world_view_transform)[:,2]);uv=np.rint(np.nan_to_num(screen,nan=-1e7)).astype('int64')
                on=(z>0)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)
                eligible=np.zeros(ns,bool);eligible[on]=allowed[uv[on,1],uv[on,0]]
            old_ids=np.load(src/'existing/contributors.npz')['removed_ids'];removed=np.zeros(ns,bool);removed[old_ids]=True;initial=removed.copy();history=[]
            for step in range(8):
                kw={**st,'opacities':st['opacities'].clone()};kw['opacities'][torch.from_numpy(removed).cuda()]=0
                out,w,c=attribution(n,cam,kw,maps);w=cpu(w);alpha=cpu(out['alpha'][0]);g=cpu(out['rgb'])-(1-alpha)[None]*cpu(n.bg).reshape(3,1,1)
                add=(w[:,0]>1e-7)&eligible&~removed
                history.append(dict(step=step,new_rows=int(add.sum()),query_alpha_mean=float(alpha[query].mean()),outside_center_remaining_rows=int(((w[:,0]>1e-7)&~eligible&~removed).sum())))
                if not add.any():break
                if step==7:break
                removed|=add;del out;gc.collect()
            white=g+(1-alpha)[None]
            # 显示范围保护：跨界Gaussian域外贡献继续取existing分支，不误改域外。
            bounded=state['native_rgb'].copy();bounded[:,allowed]=white[:,allowed]
            with torch.no_grad():env=cpu(n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1))
            native=g+(1-alpha)[None]*env
            bounded_native=state['native_rgb'].copy();bounded_native[:,allowed]=native[:,allowed]
            save(dest/'white.png',torch.from_numpy(bounded));save(dest/'native.png',torch.from_numpy(bounded_native));save(dest/'ungated_white.png',torch.from_numpy(white))
            save(dest/'gaussian.png',torch.from_numpy(g))
            from PIL import Image
            Image.fromarray(np.round(np.clip(alpha,0,1)*255).astype('uint8')).save(dest/'alpha.png')
            new=removed&~initial;assert not np.any(new&~eligible)
            np.savez_compressed(dest/'evidence.npz',new_unknown_ids=np.flatnonzero(new).astype('int32'),screen_centers=screen[new],
                existing_removed_ids=old_ids,eligible_ids=np.flatnonzero(eligible).astype('int32'),allowed=allowed,query=query,
                alpha=alpha,gaussian_rgb=g,white_rgb=bounded,native_rgb=bounded_native,environment_rgb=env,
                residual_ids=np.flatnonzero((w[:,0]>1e-7)&~removed).astype('int32'),residual_mass=w[(w[:,0]>1e-7)&~removed,0])
            rec=dict(raw=raw,history=history,new_unknown_rows=int(new.sum()),new_rows_outside_blue_centers=int((new&~eligible).sum()),
                outside_output_max_difference=float(np.abs(bounded[:,~allowed]-state['native_rgb'][:,~allowed]).max()),
                final_query_alpha_mean=float(alpha[query].mean()),final_query_alpha_max=float(alpha[query].max()),
                final_effect_alpha_mean=float(alpha[inp['user_contact_query']].mean()),
                remaining_dark_effect_pixels=int((((np.clip(bounded,0,1)*np.array([.2126,.7152,.0722])[:,None,None]).sum(0)<.2)&inp['user_contact_query']).sum()),
                query_pixels=int(query.sum()),actual_rasterizer_rerendered=True,unknown_scope='frame-specific projected Gaussian center within user blue polygon; not global semantic GT',
                outside_preservation='same-frame existing branch for out-of-scope rays; cross-border support not globally suppressed',
                white_equation='G_after+(1-alpha_after)*white, within allowed polygon',environment_checkpoint_unchanged=True,
                training_updates=0,labels_for_training=False,human_verdict=None)
            write(dest/'receipt.json',rec);records.append(rec);print('BOUNDED',json.dumps(rec),flush=True)
            del out,st,dy,w;gc.collect()
        assert n.unchanged()
    write(a.output/'summary.json',dict(records=records,wall_seconds=time.monotonic()-start,cuda_peak_reserved=torch.cuda.max_memory_reserved(),training_updates=0))

if __name__=='__main__':main()
