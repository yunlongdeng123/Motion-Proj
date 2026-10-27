"""Current-time footprint purity vs global union; same learned background layer.

Identity is not proved by a per-view target fraction. This probe diagnoses
whether global union or simultaneous mixed footprint dominates collateral damage.
"""
import json
import numpy as np
from train_actor import NativeMotion,ROOT,OLD,torch,cpu,prepare_rasterization,helpers
from train_target_unknown import Decomposition
from probe_target_unknown import CTX,save
from segmentation import write
from gpu_slot import exclusive

def main():
    R=ROOT/'target_unknown_v2';D=R/'time_local_probe';D.mkdir(exist_ok=False);ev=np.load(R/'evidence_probe/evidence.npz');frames=json.loads((ROOT/'datasets'/CTX/'manifest.json').read_text())['frames'];records=[];masks={}
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000);ns=len(n.static._xyz);model=Decomposition(n,ev,np.load(R/'clean_memory199_dense_r2/seeds.npz'),'joint');model.load_state_dict(torch.load(str(R/'dense199/joint/formal/checkpoint010000.pth'),map_location='cuda')['model'])
        for i,f in enumerate(frames):
            raw=f['raw'];e=np.load(R/'evidence_probe'/('%03d.npz'%raw));ids=e['ids'];w=e['weights'];fraction=w[:,1]/np.maximum(w[:,9],1e-8);cam=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{});st=prepare_rasterization(cam,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);st['features']=st['features'][:,:2];env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1);base=helpers.raster(cam,n.args,n.static.active_sh_degree,st,n.bg);ref=base['rgb']+(1-base['alpha'])*env;body=np.load(ROOT/'datasets'/CTX/('%03d.npz'%raw))['body'];outside=torch.from_numpy(~__import__('scipy.ndimage',fromlist=['binary_dilation']).binary_dilation(body,iterations=8)).cuda()
            for purity in [.75,.95]:
                name='purity'+str(int(purity*100));mask=np.zeros(ns,bool);mask[ids[(w[:,1]>.01)&(fraction>=purity)]]=True;masks.setdefault(name,{})[str(raw)]=np.flatnonzero(mask).tolist();kw=model.kwargs(n,cam,st);kw['opacities']=kw['opacities'].clone();kw['opacities'][:ns]=st['opacities'];kw['opacities'][torch.from_numpy(np.r_[mask,np.zeros(len(model.xyz),bool)]).cuda()]=0;kw['shs']=kw['shs'].clone();kw['shs'][:ns]=st['shs']
                r=helpers.raster(cam,n.args,n.static.active_sh_degree,kw,n.bg);rgb=r['rgb']+(1-r['alpha'])*env;delta=(rgb-ref).abs().mean(0);save(D/name/'native_rgb'/('%03d.png'%raw),rgb)
                records.append(dict(raw=raw,purity=purity,suppressed_rows=int(mask.sum()),original_static_target_mass_suppressed_fraction=float(w[np.isin(ids,np.flatnonzero(mask)),1].sum()/w[:,1].sum()),protected_changed_pixels=int((delta[outside]>1/255).sum()),protected_change_mean=float(delta[outside].mean())))
            print('TIME_LOCAL',raw,flush=True)
        assert n.unchanged()
    write(D/'masks.json',masks);write(D/'summary.json',dict(context=CTX,records=records,background_layer='frozen joint_dense199 10K,6080 rows',static_color_original=True,source_checkpoint_unchanged=True,training_updates=0,purity_not_3D_identity=True,quality_accepted=False,human_verdict=None))

if __name__=='__main__':main()
