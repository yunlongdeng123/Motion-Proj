"""Quantify shared pixel support and exposed environment, without changing models."""
import json,gc,argparse
import numpy as np
from train_actor import NativeMotion,ROOT,OLD,torch,cpu,prepare_rasterization,helpers
from probe_target_unknown import CTX,save
from segmentation import write
from gpu_slot import exclusive

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='mixed_support_audit');a=p.parse_args()
    R=ROOT/'target_unknown_v2';D=R/a.output;D.mkdir(exist_ok=False);ev=np.load(R/'evidence_probe/evidence.npz');candidate=ev['target_candidate'];fraction=ev['sum_target']/np.maximum(ev['sum_total'],1e-8);same=[]
    for p in sorted((R/'evidence_probe').glob('[0-9][0-9][0-9].npz')):
        d=np.load(p);w=d['weights'];hit=candidate[d['ids']];mixed=hit&(w[:,1]>.01)&(w[:,11]>.1);same.append(dict(raw=int(p.stem),same_frame_target_and_safe_background_rows=int(mixed.sum()),target_mass_on_same_frame_mixed_rows=float(w[mixed,1].sum()),total_target_mass=float(w[:,1].sum())))
    result=dict(target_candidate_rows=int(candidate.sum()),candidate_target_fraction_quantiles=np.quantile(fraction[candidate],[0,.1,.5,.9,1]).tolist(),candidate_all_view_target_mass_fraction=float(ev['sum_target'][candidate].sum()/ev['sum_total'][candidate].sum()),same_frame=same,whole_Gaussian_ownership_not_identifiable_from_projection_alone=True)
    records=[]
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000);ds=ROOT/'datasets'/CTX;frames=json.loads((ds/'manifest.json').read_text())['frames']
        for raw in [98,113,114]:
            i=next(i for i,f in enumerate(frames) if f['raw']==raw);f=frames[i];cam=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{});kw=prepare_rasterization(cam,n.static,n.args);kw['means2D']=torch.zeros_like(kw['means3D']);kw['opacities']=kw['opacities'].clone();kw['opacities'][torch.from_numpy(candidate).cuda()]=0;r=helpers.raster(cam,n.args,n.static.active_sh_degree,kw,n.bg);env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1);fallback=(1-r['alpha'])*env;rgb=r['rgb']+fallback
            # Saturation in the saved RGB is a diagnostic, not a missing-surface label.
            sat=(rgb>1).any(0);dom=(fallback.abs().sum(0)>r['rgb'].abs().sum(0));save(D/('raw%d_no_env.png'%raw),r['rgb']);save(D/('raw%d_env_fallback.png'%raw),fallback);save(D/('raw%d_bounded_env.png'%raw),r['rgb']+(1-r['alpha'])*env.clamp(0,1))
            records.append(dict(raw=raw,environment_rgb_max=float(env.max()),environment_rgb_min=float(env.min()),saturated_pixels=int(sat.sum()),saturated_pixels_environment_dominant=int((sat&dom).sum()),mean_remaining_alpha=float(r['alpha'].mean())))
        assert n.unchanged()
    result.update(environment=records,source_checkpoint_unchanged=True,training_updates=0);write(D/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='same_frame'}),flush=True)

if __name__=='__main__':main()
