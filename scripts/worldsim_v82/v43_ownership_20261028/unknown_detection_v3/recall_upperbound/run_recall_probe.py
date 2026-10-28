"""Diagnostic recall ceiling: fixed raw-image domains, after-removal attribution.

This is an intervention audit, NOT a detector label generator or training set.
Background impact is logged and never vetoes a proposed row. Checkpoints frozen.
"""
import argparse, gc, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_detection import ROOT, CTX, load_native, kwargs, attribution, authorized, write, save, torch, cpu, helpers, exclusive
from render_detection import maps_for

def domains(raw, ds, body_probe, shadow, contact, n, cam, expansion):
    old, _, photo, _ = maps_for(raw, ds, shadow, body_probe, n, cam, contact)
    body = old[0] > 0
    y, x = np.where(body)
    assert len(x)
    h, w = body.shape
    bw, bh = x.max()-x.min()+1, y.max()-y.min()+1
    bounds = [max(0, int(x.min()-expansion*bw)), max(0, int(y.min()-.5*expansion*bh)),
              min(w, int(x.max()+1+expansion*bw)), min(h, int(y.max()+1+expansion*bh))]
    broad = np.zeros_like(body)
    broad[bounds[1]:bounds[3], bounds[0]:bounds[2]] = True
    old_domain = old[11] > 0
    broad |= old_domain
    effect = (old[3]>0) | photo
    maps = np.stack([old_domain, broad, body, effect, ~broad, np.ones_like(body)]).astype('float32')
    return maps, bounds

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--body-probe',type=Path,required=True)
    p.add_argument('--shadow',type=Path,required=True)
    p.add_argument('--contact',type=Path,required=True)
    p.add_argument('--raws',default='98,113,114')
    p.add_argument('--expansion',type=float,default=1.)
    p.add_argument('--mass-epsilon',type=float,default=1e-4)
    p.add_argument('--max-rounds',type=int,default=5)
    a=p.parse_args(); authorized(); a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); records=[]
    channels=['old_domain','broad_domain','body','effect','outside','total']
    def status(**v):write(a.output/'status.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],**v))
    status(phase='waiting_gpu')
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz)
        actor_versions={k:(v._version,tuple(v.shape)) for k,v in actor.state_dict().items()}
        actor_path=ROOT/'actor_campaign_v1'/CTX/'formal10k/checkpoint010000.pth'
        actor_file_before=(actor_path.stat().st_size,actor_path.stat().st_mtime_ns)
        wanted=set(map(int,a.raws.split(',')))
        for i,f in enumerate(frames):
            raw=f['raw']
            if raw not in wanted:continue
            authorized();cam,st,dy=kwargs(n,actor,f,i)
            maps,bounds=domains(raw,ds,a.body_probe,a.shadow,a.contact,n,cam,a.expansion)
            destination=a.output/f'{raw:03d}';destination.mkdir()
            np.savez_compressed(destination/'domains.npz',maps=maps,bounds=bounds)
            original=np.load(ds/f'{raw:03d}.npz')['rgb'];Image.fromarray(original).save(destination/'original.png')
            overlay=original.astype(float);domain=maps[1]>0
            overlay[domain]=overlay[domain]*.75+np.array([255,170,35])*.25
            overlay[maps[3]>0]=np.array([175,90,225])
            Image.fromarray(np.uint8(np.clip(overlay,0,255))).save(destination/'domain_overlay.png')
            with torch.no_grad():
                env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                from utils.sh_utils import eval_sh
                if st.get('colors_precomp') is not None:colors=cpu(st['colors_precomp'])
                else:
                    direction=st['means3D']-cam.camera_center
                    direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-8)
                    colors=cpu((eval_sh(n.static.active_sh_degree,st['shs'].transpose(1,2),direction)+.5).clamp_min(0))
            env_cpu=cpu(env);bg=cpu(n.bg).reshape(3,1,1)
            np.savez_compressed(destination/'environment.npz',rgb=env_cpu,bg=bg,world_directions=cpu(cam.get_world_directions(False)))
            save(destination/'environment_display.png',env)
            before=np.load(a.previous/'evidence'/f'{raw:03d}.npz')
            existing=np.zeros(ns,bool);existing[before['ids'][before['static_role']>=2]]=True
            initial_weights=None;baseline_rgb=None
            def probe(name,removed,selection_source):
                nonlocal baseline_rgb
                authorized();kw={**st,'opacities':st['opacities'].clone()}
                kw['opacities'][torch.from_numpy(removed).cuda()]=0
                # This pass deliberately runs WITH grad, recomputing T after removal.
                out,weight,closure=attribution(n,cam,kw,maps)
                wgt=cpu(weight);alpha=cpu(out['alpha'][0]);raw_g=cpu(out['rgb'])
                trans=1-alpha;gs=raw_g-trans[None]*bg
                env_term=trans[None]*env_cpu;native=raw_g+env_term
                if baseline_rgb is None:baseline_rgb=native.copy()
                dest=destination/name;dest.mkdir()
                save(dest/'native.png',out['rgb'].detach()+(1-out['alpha'].detach())*env)
                save(dest/'gaussian_only.png',torch.from_numpy(gs))
                save(dest/'environment_term.png',torch.from_numpy(env_term))
                Image.fromarray(np.round(np.clip(alpha,0,1)*255).astype('uint8')).save(dest/'alpha.png')
                np.savez_compressed(dest/'maps.npz',alpha=alpha,gaussian_rgb=gs,environment_rgb=env_term,native_rgb=native)
                ids=np.flatnonzero(wgt[:,-1]>0)
                np.savez_compressed(dest/'contributors.npz',ids=ids.astype('int32'),weights=wgt[ids],removed_ids=np.flatnonzero(removed).astype('int32'),colors=colors[ids])
                integrals=wgt.T@colors
                actual=np.stack([(gs*mask).sum((1,2)) for mask in maps])
                color_error=float(np.max(np.abs(integrals-actual)/np.maximum(1,np.abs(actual))))
                assert color_error<.003,(name,color_error)
                assert not np.any(wgt[removed]>1e-8)
                regions={}
                delta=np.abs(native-baseline_rgb).mean(0)
                for j,channel in enumerate(channels):
                    mask=maps[j]>0;den=max(int(mask.sum()),1)
                    regions[channel]=dict(pixels=int(mask.sum()),alpha_mean=float(alpha[mask].sum()/den),
                        gaussian_rgb_mean=gs[:,mask].sum(1).tolist() if not mask.any() else gs[:,mask].mean(1).tolist(),
                        env_rgb_mean=env_term[:,mask].sum(1).tolist() if not mask.any() else env_term[:,mask].mean(1).tolist(),
                        negative_native_pixels=int(((native<0).any(0)&mask).sum()),negative_environment_pixels=int(((env_term<0).any(0)&mask).sum()),
                        changed_pixels=int(((delta>1/255)&mask).sum()))
                row=dict(raw=raw,stage=name,removed_static=int(removed.sum()),actor_removed=len(actor.xyz),
                    contributing_static_in_broad=int((wgt[:,1]>a.mass_epsilon).sum()),regions=regions,
                    adjoint_closure=closure,SH_integral_relative_error=color_error,selection_source=selection_source,
                    environment_unmodified=True,background_veto=False,labels_for_training=False)
                if initial_weights is not None:
                    after=wgt[:,1]>a.mass_epsilon;initial=initial_weights[:,1]>a.mass_epsilon
                    newly=after&~initial
                    row['newly_visible_vs_actor_omitted']=int(newly.sum())
                    row['newly_visible_broad_alpha_mass']=float(wgt[newly,1].sum())
                write(dest/'receipt.json',row);records.append(row)
                print('RECALL',raw,name,'rows',row['removed_static'],'alpha',round(regions['body']['alpha_mean'],6),'effect',round(regions['effect']['alpha_mean'],6),'env_negative',regions['body']['negative_environment_pixels'],flush=True)
                status(phase='running',raw=raw,stage=name,wall_seconds=time.monotonic()-start)
                del out,weight,kw;gc.collect()
                return wgt,row
            initial_weights,_=probe('actor_omitted',np.zeros(ns,bool),'existing actor identity only')
            np.savez_compressed(destination/'initial_weights.npz',ids=np.arange(ns,dtype='int32'),weights=initial_weights)
            probe('existing',existing,'previous frozen candidate labels')
            old_open=existing|(initial_weights[:,0]>a.mass_epsilon)
            probe('gates_open_same_domain',old_open,'all initial old-domain contributors; no depth/history/fraction gates')
            current=existing|(initial_weights[:,1]>a.mass_epsilon)
            wgt,row=probe('broad_once',current,'fixed raw-body-relative broad domain; all initial contributors')
            for step in range(1,a.max_rounds+1):
                epsilon=a.mass_epsilon if step==1 else 1e-7
                added=(wgt[:,1]>epsilon)&~current
                if not added.any():break
                current|=added
                wgt,row=probe(f'peel_{step:02d}',current,'recomputed post-removal contribution in FIXED domain; diagnostic only')
                if row['regions']['broad_domain']['alpha_mean']<1e-6:break
            write(destination/'final.json',dict(final_stage=row['stage'],removed_static=int(current.sum()),domain_bounds=bounds,
                remaining_gaussian_alpha_mean=row['regions']['broad_domain']['alpha_mean'],
                environment_negative_pixels=row['regions']['broad_domain']['negative_environment_pixels'],
                Gaussian_extinction_is_not_semantic_recall=True,background_damage_not_veto=True))
            del st,dy,initial_weights,wgt;gc.collect()
        assert n.unchanged()
        assert actor_versions=={k:(v._version,tuple(v.shape)) for k,v in actor.state_dict().items()}
        assert actor_file_before==(actor_path.stat().st_size,actor_path.stat().st_mtime_ns)
        write(a.output/'summary.json',dict(schema='target-recall-ceiling-v1',records=records,channels=channels,
            raw_domain_expansion=a.expansion,mass_epsilon=a.mass_epsilon,training_updates=0,checkpoint_frozen=True,
            background_protection_enabled=False,intervention_weights_used_only_for_diagnostic_peeling=True,
            semantic_labels_modified=False,environment_modified=False,inpainting=False,colors_modified=False,
            independent_recall_metric=False,human_verdict=None,actor_checkpoint_and_tensors_unchanged=True,
            static_rows=ns,static_row_namespace='original_static_rows',actor_removed_all_stages=True,
            initial_weights_dense=True,stage_omitted_rows_are_zero=True,stage_export_predicate='total_mass > 0',
            previous_evidence=str(a.previous/'evidence'),body_probe=str(a.body_probe)))
        status(phase='complete',wall_seconds=time.monotonic()-start,cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()
