"""Exact alpha*T adjoint of frozen native rasterization, all Gaussian rows.

Per-Gaussian semantic evidence is the derivative with respect to auxiliary
features. Geometry, original branch, opacity, and RGB are never optimized here.
2D semantics are predictions, not geometric or semantic ground truth.
"""
import argparse,json,os,sys,time,gc,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
ROOT=OLD.parent/'v43_ownership_20261028'
sys.path.insert(0,str(OLD/'code'))
from render_motion_suite import NativeMotion,prepare_rasterization,helpers,torch,cpu
from gpu_slot import exclusive
from segmentation import CONTEXTS,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--segmentation',default='segformer_b0_attempt3');p.add_argument('--attempt',default='attribution_v1');p.add_argument('--single-frame',action='store_true');a=p.parse_args()
    out=ROOT/a.attempt/a.context;out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],context=a.context,phase='waiting_GPU',frames=0,training_updates=0)
    def update(**kw):status.update(kw);write(out/'status.json',status)
    update()
    with exclusive():
        torch.set_num_threads(1);update(phase='loading_checkpoint')
        run=OLD/'native_scene_training_v1'/a.context/'formal30k';n=NativeMotion(run,30000)
        source=OLD/'motion_scene_inputs_v1'/a.context;sensor=json.loads((source/'sensor_preparation.json').read_text())
        sizes=[len(n.static._xyz),len(n.dynamic._xyz)];N=sum(sizes)
        sums=torch.zeros((N,22),device='cuda');support=torch.zeros((N,20),device='cuda',dtype=torch.int16);target_frames=torch.zeros(N,device='cuda',dtype=torch.int16)
        selected=[24] if a.single_frame else range(49);records=[]
        for i in selected:
            f=sensor['frames'][i];raw=f['raw'];row=dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=i*.02-.48)
            camera=n.camera(row,{});sem=np.load(ROOT/a.segmentation/a.context/('%03d.npz'%raw));labels=sem['labels'];confidence=sem['confidence'].astype(np.float32)
            target=np.array(Image.open(source/'dynamic_actor_assets/actor_0/body_masks'/(f['frame_id']+'.png')).resize((480,320),Image.Resampling.NEAREST))>0
            interior=binary_erosion(target,iterations=2);agreement=interior&np.isin(labels,[14,15,16,17])
            maps=np.stack([(labels==k).astype(np.float32)*confidence for k in range(20)]+[target.astype(np.float32),np.ones_like(confidence)])
            with torch.no_grad():
                s=prepare_rasterization(camera,n.static,n.args);d=prepare_rasterization(camera,n.dynamic,n.args,dynamic=True)
                for kw in [s,d]:kw['means2D']=torch.zeros_like(kw['means3D'])
                kw=helpers.merge_rows(s,d)
            # Native CUDA MAX_FEATURES=24 includes the 2 depth/alpha channels.
            feat=torch.zeros((N,22),device='cuda',requires_grad=True)
            kw['features']=torch.cat((kw['features'],feat),1)
            assert kw['features'].shape[1]<=24
            result=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg)
            gradient_maps=torch.from_numpy(maps).cuda()
            weights=torch.autograd.grad((result['extra']*gradient_maps).sum(),feat)[0]
            assert torch.isfinite(weights).all() and float(weights.min())>=-1e-5
            weights=weights.clamp_min(0)
            alpha=result['alpha'][0].detach();closure=float((weights[:,21].sum()-alpha.sum()).abs()/alpha.sum().clamp_min(1))
            target_closure=float((weights[:,20].sum()-(alpha*gradient_maps[20]).sum()).abs()/weights[:,20].sum().clamp_min(1))
            assert closure<2e-4 and target_closure<2e-4,(closure,target_closure)
            total=weights[:,21];class_mass,best=weights[:,:20].max(1);purity=class_mass/total.clamp_min(1e-8)
            accepted=(total>.01)&(purity>.65)&(best>0)
            semids=torch.where(accepted,best,torch.zeros_like(best))
            target_frac=weights[:,20]/total.clamp_min(1e-8);vehicle=weights[:,14:18].sum(1)/total.clamp_min(1e-8)
            sums+=weights;support[torch.arange(N,device='cuda'),semids]+=accepted.to(torch.int16)
            target_frames+=((total>.01)&(target_frac>.8)&(vehicle>.6)).to(torch.int16)
            active=total>.001;ids=torch.nonzero(active).flatten()
            np.savez_compressed(out/('%03d.npz'%raw),ids=cpu(ids).astype('int32'),class_id=cpu(semids[ids]).astype('uint8'),mass=cpu(total[ids]).astype('float16'),purity=cpu(purity[ids]).astype('float16'),target_fraction=cpu(target_frac[ids]).astype('float16'),vehicle_fraction=cpu(vehicle[ids]).astype('float16'))
            record=dict(raw=raw,active_contributing_rows=int(active.sum()),alpha_sum=float(alpha.sum()),adjoint_closure_relative=closure,target_adjoint_closure_relative=target_closure,static_target_alpha_mass=float(weights[:sizes[0],20].sum()),dynamic_target_alpha_mass=float(weights[sizes[0]:,20].sum()),static_vehicle_alpha_mass=float(weights[:sizes[0],14:18].sum()),semantic_candidate_rows=int(accepted.sum()))
            records.append(record);update(phase='attributing',frames=len(records),last_raw=raw)
            print('ATTRIBUTE',a.context,raw,record['active_contributing_rows'],round(time.monotonic()-started,2),flush=True)
            del result,weights,feat,kw,s,d;gc.collect()
        np.savez_compressed(out/'aggregate.npz',mass=cpu(sums),support_frames=cpu(support),target_support_frames=cpu(target_frames),branch_sizes=np.array(sizes))
        assert n.unchanged()
        write(out/'summary.json',dict(context=a.context,branch_sizes=sizes,records=records,semantic_map='cityscapes19-plus-unknown-v1',semantic_source=a.segmentation,weight_definition='exact native alpha*T via auxiliary feature adjoint, confidence-weighted 2D predictions',all_Gaussian_rows=True,original_checkpoint_unchanged=True,quality_accepted=False,human_verdict=None,not_semantic_ground_truth=True))
        update(phase='complete',wall_seconds=time.monotonic()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
if __name__=='__main__':main()
