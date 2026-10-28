"""Frozen native contributor audit. Body evidence precedes shadow evidence.

No training, no generated teacher, no dependence on deletion output labels.
Full and actor-omitted transmittance are separate mechanical observations.
"""
import os,sys,json,time,argparse,gc
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion,distance_transform_edt
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
sys.path.insert(0,str(ROOT/'code'))
from train_actor import Actor,NativeMotion,OLD,prepare_rasterization,helpers,torch,cpu
from gpu_slot import exclusive
CTX='14705303724557273004_3105_000_3125_000'

def authorized():
    a=json.loads((ROOT/'authorization.json').read_text())
    assert a.get('allow_target_detection') and CTX in a.get('detection_contexts',[])
    assert not a.get('gpu_execution_hold') and not a.get('user_research_hold') and not (ROOT/'USER_STOP').exists()

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.round(np.clip(cpu(x).transpose(1,2,0),0,1)*255).astype('uint8')).save(p)

def load_native():
    ds=ROOT/'datasets'/CTX;frames=json.loads((ds/'manifest.json').read_text())['frames']
    actor=Actor(np.load(ds/'seeds.npz'))
    ck=ROOT/'actor_campaign_v1'/CTX/'formal10k/checkpoint010000.pth'
    actor.load_state_dict(torch.load(str(ck),map_location='cuda')['actor']);actor.requires_grad_(False)
    n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000)
    return n,actor,ds,frames

def kwargs(n,actor,f,i):
    cam=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{})
    with torch.no_grad():
        st=prepare_rasterization(cam,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);st['features']=st['features'][:,:2]
        dy=actor.kwargs(n,cam,f);dy['features']=dy['features'][:,:2]
    return cam,st,dy

def attribution(n,cam,kw,maps):
    # Auxiliary-feature adjoint is exactly the renderer's alpha*T coefficient.
    aux=torch.zeros((len(kw['means3D']),len(maps)),device='cuda',requires_grad=True)
    inp={**kw,'features':torch.cat((kw['features'][:,:2],aux),1)}
    out=helpers.raster(cam,n.args,n.static.active_sh_degree,inp,n.bg)
    w=torch.autograd.grad((out['extra']*torch.from_numpy(maps).cuda()).sum(),aux)[0].clamp_min(0)
    closure=float((w[:,-1].sum()-out['alpha'].sum()).abs()/out['alpha'].sum().clamp_min(1))
    assert closure<3e-4,closure
    return out,w,closure

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--raws',default='all');p.add_argument('--resume',action='store_true');a=p.parse_args()
    authorized();a.output.mkdir(parents=True,exist_ok=a.resume);started=time.monotonic()
    def status(**kw):write(a.output/'status.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],training_updates=0,**kw))
    status(phase='waiting_gpu')
    with exclusive():
        torch.set_num_threads(1);n,actor,ds,frames=load_native();ns=len(n.static._xyz);na=len(actor.xyz)
        rows=[];wanted=None if a.raws=='all' else set(map(int,a.raws.split(',')))
        status(phase='native_contributor_audit',static_rows=ns,actor_rows=na)
        for i,f in enumerate(frames):
            if wanted is not None and f['raw'] not in wanted:continue
            raw=f['raw'];dest=a.output/('%03d.npz'%raw)
            if a.resume and dest.exists():continue
            authorized();ob=np.load(ds/('%03d.npz'%raw));body=ob['body'].astype(bool)
            radius=int(np.clip(round(.04*np.sqrt(body.sum())),1,5));core=binary_erosion(body,iterations=radius)
            distance=distance_transform_edt(~body);ring=(distance<=radius)&~core
            sem=np.load(ROOT/'segformer_b0_attempt3'/CTX/('%03d.npz'%raw));road=(sem['labels']==1)&(sem['confidence']>=.8)&~body
            depth=np.nan_to_num(ob['depth'],posinf=0,neginf=0);valid=np.isfinite(ob['depth'])&body
            maps=np.stack([body,core,ring,road,valid,depth*valid,depth*depth*valid,np.ones_like(body)]).astype('float32')
            cam,st,dy=kwargs(n,actor,f,i);allkw=helpers.merge_rows(st,dy)
            both=[];closures=[]
            for name,kw in [('full',allkw),('without_actor',st)]:
                out,w,closure=attribution(n,cam,kw,maps);both.append(cpu(w));closures.append(closure)
                with torch.no_grad():env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                save(a.output/name/'native_rgb'/('%03d.png'%raw),out['rgb'].detach()+(1-out['alpha'].detach())*env)
                np.savez_compressed(a.output/name/('%03d_maps.npz'%raw),alpha=cpu(out['alpha'][0]).astype('float16'))
                del out,w;gc.collect()
            full,omit=both;active=(omit[:,0]>.0001)|(omit[:,2]>.0001)|(full[:ns,0]>.0001)
            ids=np.flatnonzero(active)
            with torch.no_grad():
                z=(torch.cat((st['means3D'],torch.ones_like(st['means3D'][:,:1])),1)@cam.world_view_transform[:,2:3])[:,0]/n.scale
                xyz=cpu(st['means3D']);Ti=np.linalg.inv(n.T);world=xyz@Ti[:3,:3].T+Ti[:3,3]
                sigma=cpu(st['scales'].amax(1))/n.scale
            np.savez_compressed(dest,ids=ids.astype('int32'),full=full[ids],without_actor=omit[ids],actor=full[ns:],center_z_m=cpu(z)[ids],world=world[ids].astype('float32'),sigma_m=sigma[ids],core=core,ring=ring,body=body)
            record=dict(raw=raw,rows=len(ids),body_pixels=int(body.sum()),radius_px=radius,adjoint_closure=closures,static_body_mass_full=float(full[:ns,0].sum()),static_body_mass_without_actor=float(omit[:,0].sum()),actor_body_mass=float(full[ns:,0].sum()))
            write(a.output/('%03d_record.json'%raw),record);rows.append(record)
            status(phase='native_contributor_audit',frames_completed=len(list(a.output.glob('*_record.json'))),last_raw=raw,wall_seconds=time.monotonic()-started)
            print('DETECTION_PROBE',json.dumps(record),flush=True);del allkw,st,dy,both;gc.collect()
        assert n.unchanged()
        records=[json.loads(p.read_text()) for p in sorted(a.output.glob('*_record.json'))]
        write(a.output/'summary.json',dict(schema='target-detection-frozen-contribution-v1',context=CTX,static_rows=ns,actor_rows=na,frames=records,channels=['body','core','ring','road_semantic_only','target_depth_valid','target_depth_sum','target_depth_squared','total'],source_checkpoint_unchanged=True,training_updates=0,road_semantics_not_background_proof=True,actor_identity_is_existing_asset_not_GT=True,human_verdict=None,quality_accepted=False))
        status(phase='complete',frames_completed=len(records),wall_seconds=time.monotonic()-started,cuda_peak_reserved=torch.cuda.max_memory_reserved())

if __name__=='__main__':main()
