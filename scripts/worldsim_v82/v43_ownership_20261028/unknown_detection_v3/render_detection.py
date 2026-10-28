"""Native rendering of fixed target-specific labels, with alpha/effect audits."""
import os,json,time,gc,argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from probe_detection import ROOT,CTX,load_native,kwargs,attribution,authorized,write,save,torch,cpu,helpers,exclusive
from classify_detection import classify,removal,CHANNELS,REASONS,CONFIG

def maps_for(raw,ds,shadow,body_probe,n,cam,contact=None):
    ob=np.load(ds/('%03d.npz'%raw));p=np.load(body_probe/('%03d.npz'%raw));s=np.load(shadow/('%03d.npz'%raw))
    body=ob['body'].astype(bool);strict=s['shadow_candidate_strict' if 'shadow_candidate_strict' in s.files else 'shadow_candidate'].astype(bool)
    # Same original-input evidence contract as shadow detector. Missing motion
    # is not negative evidence; retain photo-supported candidates separately.
    photo=s['shadow_evidence_valid']&(s['luminance_ratio']<=.84)&(s['luminance_drop']>=.035)&(s['attenuation_rgb_residual']<=.035)&~body
    if 'photo_supported_unknown' in s.files:photo|=s['photo_supported_unknown'].astype(bool)
    if 'shadow_propagated_candidate' in s.files:photo|=s['shadow_propagated_candidate'].astype(bool)
    if contact is not None:
        contact_data=np.load(contact/('%03d.npz'%raw))
        photo|=contact_data['contact_ground_photo_effect'].astype(bool)
    bg=s['road_counterevidence_valid'].astype(bool);world=s['world_point'];valid=np.isfinite(world).all(-1)
    # Use float64 for large global coordinates before casting to model units.
    model=np.nan_to_num(world).reshape(-1,3)@n.T[:3,:3].T+n.T[:3,3]
    view=cpu(cam.world_view_transform).astype('float64');z=(np.column_stack((model,np.ones(len(model))))@view[:,2])/n.scale;z=z.reshape(body.shape)
    bg&=valid;z=np.where(bg,z,0);depth=np.nan_to_num(ob['depth'],posinf=0,neginf=0);tv=np.isfinite(ob['depth'])&body
    union=body|p['ring']|photo|strict
    maps=np.stack([body,p['core'],p['ring'],strict,photo,bg,z,z*z,tv,depth*tv,depth*depth*tv,union,np.ones_like(body)]).astype('float32')
    return maps,s,photo,bg

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--shadow',type=Path,required=True);p.add_argument('--body-probe',type=Path,required=True);p.add_argument('--contact',type=Path);p.add_argument('--raws',default='98,113,114');p.add_argument('--profile',choices=['strict','photo','shadow'],default='photo');a=p.parse_args()
    authorized();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def status(**x):write(a.output/'status.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],training_updates=0,**x))
    status(phase='waiting_gpu')
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz);na=len(actor.xyz)
        body_counts=np.zeros(ns+na,np.uint16)
        for path in sorted(a.body_probe.glob('[0-9][0-9][0-9].npz')):
            e=np.load(path);q=e['without_actor'];hit=(q[:,0]>.02)&(q[:,0]/np.maximum(q[:,-1],1e-8)>.05);body_counts[e['ids'][hit]]+=1
        body_counts[ns:]=len(frames)
        write(a.output/'contract.json',dict(schema='target-detection-native-v1',context=CTX,profile=a.profile,configuration=CONFIG,reasons=REASONS,channels=CHANNELS,shadow_source=str(a.shadow),contact_effect_source=str(a.contact) if a.contact else None,body_probe=str(a.body_probe),source_checkpoint_frozen=True,training_updates=0,body_history_used_all_source_frames=True,inpainting_used=False,labels_depend_on_edited_RGB=False))
        wanted=None if a.raws=='all' else set(map(int,a.raws.split(',')));records=[]
        for i,f in enumerate(frames):
            raw=f['raw']
            if wanted is not None and raw not in wanted:continue
            authorized();cam,st,dy=kwargs(n,actor,f,i);maps,shadow,photo,bg=maps_for(raw,ds,a.shadow,a.body_probe,n,cam,a.contact)
            # Labels always use the same actor-omitted frozen probe, regardless
            # of the removal mode to be evaluated below.
            out,w,c=attribution(n,cam,st,maps);w=cpu(w);del out
            af=np.load(a.body_probe/('%03d.npz'%raw))['actor'];wa=np.zeros((na,len(maps)),np.float32);wa[:,:3]=af[:,:3];wa[:,-1]=af[:,-1]
            weight=np.concatenate((w,wa),0)
            with torch.no_grad():
                allkw=helpers.merge_rows(st,dy);z=(torch.cat((allkw['means3D'],torch.ones_like(allkw['means3D'][:,:1])),1)@cam.world_view_transform[:,2:3])[:,0]/n.scale
            labels=classify(weight,cpu(z),body_counts,ns,a.profile);role=labels['role'];ids=np.flatnonzero((role[:ns]>0)|(w[:,5]>.02));evidence=a.output/'evidence';evidence.mkdir(exist_ok=True)
            np.savez_compressed(evidence/('%03d.npz'%raw),ids=ids.astype('int32'),static_role=role[ids],reason_bits=labels['reason_bits'][ids],body_mass=w[ids,0],shadow_mass=w[ids,3 if a.profile=='strict' else 4],bg_mass=w[ids,5],total_mass=w[ids,-1],target_fraction=w[ids,0]/np.maximum(w[ids,-1],1e-8),shadow_fraction=w[ids,3 if a.profile=='strict' else 4]/np.maximum(w[ids,-1],1e-8),bg_protected=labels['background_protected'][ids],bg_weak=labels['bg_weak'][ids],bg_depth_rms=labels['bg_rms'][ids],shadow_effect=labels['shadow_effect'][ids],body_frames=body_counts[ids],pixel_role=np.where(maps[0]>0,1,np.where(maps[3]>0,3,np.where(photo,3,np.where(maps[2]>0,2,0)))).astype('uint8'),pixel_shadow_effect=photo.astype('uint8'))
            src=np.load(ds/('%03d.npz'%raw))['rgb'];Image.fromarray(src).save(a.output/('%03d_original.png'%raw));overlay=src.astype(float);overlay[maps[0]>0]=overlay[maps[0]>0]*.65+np.array([0,220,140])*.35;overlay[photo]=overlay[photo]*.4+np.array([184,100,245])*.6;Image.fromarray(np.round(overlay).astype('uint8')).save(a.output/('%03d_overlay.png'%raw))
            target_domain=(maps[0]>0)|photo;protected_outside=~binary_dilation(target_domain,iterations=2)
            env=None;baseline=None;basealpha=None
            with torch.no_grad():
                env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                for mode in ['baseline','remove_car','remove_u','remove_car_u']:
                    mask=removal(role,mode);kw={**allkw,'opacities':allkw['opacities'].clone()};kw['opacities'][torch.from_numpy(mask).cuda()]=0
                    result=helpers.raster(cam,n.args,n.static.active_sh_degree,kw,n.bg);rgb=result['rgb']+(1-result['alpha'])*env
                    save(a.output/mode/'native_rgb'/('%03d.png'%raw),rgb)
                    arr=cpu(rgb);alpha=cpu(result['alpha'][0]);delta=np.zeros_like(alpha)
                    if baseline is None:baseline=arr.copy();basealpha=alpha.copy()
                    else:delta=np.abs(arr-baseline).mean(0)
                    np.savez_compressed(a.output/mode/('%03d_maps.npz'%raw),alpha=alpha.astype('float16'),unclipped_rgb=arr.astype('float16'))
                    records.append(dict(raw=raw,mode=mode,removed_static_rows=int(mask[:ns].sum()),removed_actor_rows=int(mask[ns:].sum()),role_counts={str(k):int((role[:ns]==k).sum()) for k in range(5)},shadow_strict_pixels=int(maps[3].sum()),shadow_photo_pixels=int(photo.sum()),body_mask_pixels=int(maps[0].sum()),outside_domain_changed_pixels=int((delta[protected_outside]>1/255).sum()),outside_domain_rgb_change=float(delta[protected_outside].mean()),outside_domain_alpha_drop=float(np.maximum(basealpha-alpha,0)[protected_outside].mean()),saturated_pixels=int((arr>1).any(0).sum()),adjoint_closure=c))
                # Render effect mass on the unedited frozen full model. This is
                # a contribution map, not an original RGB texture backplate.
                aux=allkw['features'].new_zeros((ns+na,5));aux[torch.arange(ns+na,device='cuda'),torch.from_numpy(role.astype('int64')).cuda()]=1
                extra={**allkw,'features':torch.cat((allkw['features'][:,:2],aux),1)};out=helpers.raster(cam,n.args,n.static.active_sh_degree,extra,n.bg)
                np.savez_compressed(evidence/('%03d_contributions.npz'%raw),role_contribution=cpu(out['extra']).astype('float16'))
            status(phase='rendering',frames_completed=len(records)//4,last_raw=raw,wall_seconds=time.monotonic()-start)
            print('DETECTION_RENDER',raw,records[-1]['role_counts'],'photo_pixels',int(photo.sum()),'outside_changed',records[-1]['outside_domain_changed_pixels'],flush=True)
            del st,dy,allkw,w,weight;gc.collect()
        assert n.unchanged()
        write(a.output/'summary.json',dict(schema='target-detection-native-v1',context=CTX,profile=a.profile,records=records,source_checkpoint_unchanged=True,actor_source='frozen V43 measured Actor10K',training_updates=0,shadow_effect_not_car_semantics=True,certified_background_rows=0,background_counterevidence_is_weak=True,quality_accepted=False,human_verdict=None,independent_precision_recall_available=False))
        status(phase='complete',frames_completed=len(records)//4,wall_seconds=time.monotonic()-start,cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()
