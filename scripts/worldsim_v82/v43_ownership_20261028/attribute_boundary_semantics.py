"""Original/V43 row namespaces, real observed semantics and explicit contour uncertainty.

No training or renderer owner changes. The 3/5/8 pixel bands are diagnostic
abstention priors, not learned probabilities or semantic ground truth.
"""
import argparse,json,os,time,gc
import numpy as np
from scipy.ndimage import distance_transform_edt
from train_actor import Actor,NativeMotion,prepare_rasterization,helpers,torch,cpu,ROOT,OLD
from segmentation import CONTEXTS,write
from v43_resource_control import require_start
from gpu_slot import exclusive

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',default=CONTEXTS[0]);p.add_argument('--output',default='boundary_semantics_v1');p.add_argument('--single-frame',action='store_true');a=p.parse_args();require_start(a.context)
    out=ROOT/a.output/a.context;out.mkdir(parents=True,exist_ok=False);status=dict(pid=os.getpid(),start_ticks=__import__('pathlib').Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',frames=0,training_updates=0)
    def update(**kw):status.update(kw);write(out/'status.json',status)
    update()
    with exclusive():
        require_start(a.context);torch.set_num_threads(1);started=time.monotonic();dataset=ROOT/'datasets'/a.context;frames=json.loads((dataset/'manifest.json').read_text())['frames'];actor=Actor(np.load(dataset/'seeds.npz'));s=torch.load(str(ROOT/'actor_campaign_v1'/a.context/'formal10k/checkpoint010000.pth'),map_location='cuda');actor.load_state_dict(s['actor']);del s
        n=NativeMotion(OLD/'native_scene_training_v1'/a.context/'formal30k',30000);ns=len(n.static._xyz);records=[];sizes=dict(original=[ns,len(n.dynamic._xyz)],actor10k=[ns,len(actor.xyz)])
        for i in ([24] if a.single_frame else range(49)):
            f=frames[i];raw=f['raw'];ob=np.load(dataset/('%03d.npz'%raw));body=ob['body'];sem=np.load(ROOT/'segformer_b0_attempt3'/a.context/('%03d.npz'%raw));labels=sem['labels'];confidence=sem['confidence'].astype(np.float32)
            vehicle=np.isin(labels,np.arange(14,20))|body;distance=np.where(vehicle,distance_transform_edt(vehicle),distance_transform_edt(~vehicle));bands=np.stack([distance<=r for r in [3,5,8]])
            np.savez_compressed(out/('image_%03d.npz'%raw),semantic_image=labels,confidence_image=sem['confidence'],vehicle=vehicle,bands=bands)
            camera=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{});fr=[]
            for variant in ['original','actor10k']:
                with torch.no_grad():
                    st=prepare_rasterization(camera,n.static,n.args);dy=prepare_rasterization(camera,n.dynamic,n.args,dynamic=True) if variant=='original' else actor.kwargs(n,camera,f)
                    for kw in [st,dy]:kw['means2D']=torch.zeros_like(kw['means3D']);kw['features']=kw['features'][:,:2]
                    kw=helpers.merge_rows(st,dy)
                N=sum(sizes[variant]);feat=torch.zeros((N,22),device='cuda',requires_grad=True);kw['features']=torch.cat((kw['features'],feat),1)
                maps=np.stack([(labels==k).astype(np.float32)*confidence for k in range(20)]+[body.astype(np.float32),np.ones((320,480),np.float32)])
                result=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg);weights=torch.autograd.grad((result['extra']*torch.from_numpy(maps).cuda()).sum(),feat)[0].clamp_min(0)
                total=weights[:,21];closure=float((total.sum()-result['alpha'].sum()).abs()/total.sum().clamp_min(1));assert closure<2e-4
                best,classid=weights[:,:20].max(1);purity=best/total.clamp_min(1e-8);classid=torch.where((total>.01)&(purity>.65),classid,0);target=weights[:,20]/total.clamp_min(1e-8);vehicle_mass=weights[:,14:20].sum(1)/total.clamp_min(1e-8)
                del result,feat
                base=kw['features'][:,:2].detach();aux=torch.zeros((N,4),device='cuda',requires_grad=True);kw['features']=torch.cat((base,aux),1)
                result=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg);maps2=np.concatenate((bands.astype(np.float32),np.ones((1,320,480),np.float32)))
                wb=torch.autograd.grad((result['extra']*torch.from_numpy(maps2).cuda()).sum(),aux)[0].clamp_min(0);bandfrac=wb[:,:3]/wb[:,3:4].clamp_min(1e-8);row_closure=float((wb[:,3]-total).abs().max());row_relative_L1=float((wb[:,3]-total).abs().sum()/total.sum().clamp_min(1));band_closure=float((wb[:,3].sum()-result['alpha'].sum()).abs()/wb[:,3].sum().clamp_min(1));print('ADJOINT_CLOSURE',variant,raw,row_closure,row_relative_L1,band_closure,flush=True);assert row_relative_L1<2e-4 and band_closure<2e-4
                ids=torch.nonzero(total>.001).flatten();data=dict(ids=cpu(ids).astype('int32'),class_id=cpu(classid[ids]).astype('uint8'),purity=cpu(purity[ids]).astype('float16'),mass=cpu(total[ids]).astype('float16'),target_fraction=cpu(target[ids]).astype('float16'),vehicle_fraction=cpu(vehicle_mass[ids]).astype('float16'),boundary_fraction=cpu(bandfrac[ids]).astype('float16'))
                np.savez_compressed(out/('%s_%03d.npz'%(variant,raw)),**data)
                sensitivity={}
                for j,r in enumerate([3,5,8]):
                    for threshold in [.1,.2,.5]:
                        abstain=(bandfrac[:,j]>=threshold)|(classid==0);sensitivity[str(r)+'px_mass'+str(threshold)]=dict(unknown_contribution_mass_fraction=float(total[abstain].sum()/total.sum().clamp_min(1)),unknown_contributing_rows=int((abstain&(total>.001)).sum()))
                statics_car=(classid[:ns]>=14)&(classid[:ns]<=19)&(target[:ns]>.5)&(total[:ns]>.01)
                fr.append(dict(variant=variant,active_rows=len(ids),adjoint_closure=closure,band_row_closure_max=row_closure,band_row_relative_L1=row_relative_L1,band_forward_closure=band_closure,source_unknown_pixel_fraction=float((labels==0).mean()),boundary5_pixel_fraction=float(bands[1].mean()),static_vehicle_target_candidate_rows=int(statics_car.sum()),sensitivity=sensitivity))
                del kw,st,dy,weights,total,wb,aux;gc.collect()
            records.append(dict(raw=raw,variants=fr));update(phase='attributing',frames=len(records),last_raw=raw);print('BOUNDARY',raw,flush=True)
        assert n.unchanged();write(out/'summary.json',dict(context=a.context,records=records,branch_sizes=sizes,pixel_band_radii=[3,5,8],default_band_px=5,default_band_contribution_fraction=.2,unknown_rule='source confidence/purity abstention OR Gaussian visible alpha*T fraction in vehicle inner/outer contour band >= threshold',pixel_units='480x320 observed RGB, Euclidean pixel distance',instance_owner_and_motion_unchanged=True,original_checkpoint_unchanged=True,training_updates=0,quality_accepted=False,human_verdict=None));update(phase='complete',wall_seconds=time.monotonic()-started)
if __name__=='__main__':main()
