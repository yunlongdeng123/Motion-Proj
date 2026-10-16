"""True raster alpha*T attribution across49 views; conservative global fraction.

Gradients of two per-point feature channels provide road-authority mass and total
visible mass from the identical full static depth sort. Three rounds expose
hidden old layers. No original parameters are modified or trained.
"""
import os,sys,time,json,argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_surface import SurfaceExperiment,NEW
from run_experiments import write,now
import renderer

def wait():
    start=time.time()
    while time.time()-start<4*3600:
        p=NEW/'repair_follow_status.json'
        if p.exists():
            row=json.loads(p.read_text())
            if row['state']=='complete':return
            if row['state']=='evaluation_failed':raise SystemExit('Repair evaluation failed; preserve state')
        time.sleep(30)
    raise SystemExit('Attribution wait deadline')

def run():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--output-name',default='contribution_audit');p.add_argument('--overrides');a=p.parse_args()
    if a.wait:wait()
    args=argparse.Namespace(output=str(NEW/a.output_name),steps=0,warmup=1,restore=str(NEW/'repair_union_v1/final.pt'),probe=True,guard_overrides=a.overrides)
    e=SurfaceExperiment(args)
    try:
        with np.load(NEW/'away_reference_audit.npz') as f:far=torch.tensor(f['far_consistent'],device='cuda').float()
        image_far=far.reshape(1,1,int(e.surface.shape[1]),int(e.surface.shape[0]))
        masks=[]
        for i,cam in enumerate(e.cameras):
            q,conf=e.road_pixels(cam);chart=e.surface.chart(q)[None,None]
            confirmed=F.grid_sample(image_far,chart,align_corners=False)[0,0,0]>.99
            authority=(conf>.99)&confirmed&~e.surface_data['protection'][i].reshape(-1)
            road=torch.zeros((320,480),device='cuda',dtype=torch.bool)
            take=e.surface_data['clean'][i]|e.surface_data['shadow'][i]
            uv=e.surface_data['uv'][i,take];u=uv[:,0].round().long();v=uv[:,1].round().long()
            good=(u>=0)&(u<480)&(v>=0)&(v<320);road[v[good],u[good]]=True
            road=F.max_pool2d(road.float()[None,None],3,1,1)[0,0].bool()
            body=F.max_pool2d(e.body[i].any(0).float()[None,None],3,1,1)[0,0].bool()
            masks.append((authority.reshape(320,480)&(road|body)).float())
        # Re-evaluate the formerly unverified union extras by actual mass. They
        # are not automatically inherited merely because a kernel touched road.
        from rigid_model import quaternion_to_matrix
        anchor=e.surface.anchor_state;h=(anchor['means3D']@e.surface.plane[:3]+e.surface.plane[3])/e.surface.scale
        rot=quaternion_to_matrix(anchor['rotations'])
        sigma=((rot.transpose(1,2)@e.surface.plane[:3])*anchor['scales']).norm(dim=1)/e.surface.scale
        dark=anchor['colors_precomp'].amax(1)<.05
        removed=(h<.20)|((h.abs()<.20)&(3*sigma<.30))|(dark&(h.abs()<.40)&(3*sigma<.50))
        rounds=[];n=len(e.static.get_xyz)
        write(e.out/'base_selection.json',{'geometric_base_points':int(removed.sum()),'unverified_union_extras_inherited':False})
        feature=torch.zeros(n,2,device='cuda',requires_grad=True)
        current_i=0;mode='current'
        def prepare(cam,pc,pipe,scaling_modifier=1.,other=None,mask=None):
            if mode=='current':return e.original_prepare(cam,pc,pipe,scaling_modifier,feature,~removed)
            result=dict(e.surface.anchor_state)
            result['colors_precomp']=renderer.evaluate_sh_rgb(pc.get_features,F.normalize(result['means3D']-cam.camera_center,dim=-1),int(pc.active_sh_degree))
            result['mask']=result['mask']&~removed;result['other']=feature
            return result
        renderer.prepare_static=prepare;start=time.time()
        for peel in range(3):
            numerator=torch.zeros(n,device='cuda');denominator=torch.zeros(n,device='cuda')
            for mode in ['current','frozen']:
                for i,cam in enumerate(e.cameras):
                    if i%10==0:e.priority()
                    feature.grad=None
                    pkg=renderer.render(cam,e.static,None,e.config,e.bg,env_map=e.env,max_features=10,render_separate=False,is_training=False)
                    loss=(pkg['feature'][0]*masks[i]).sum()+pkg['feature'][1].sum();loss.backward()
                    g=feature.grad.detach();assert torch.isfinite(g).all()
                    numerator.add_(g[:,0]);denominator.add_(g[:,1])
                    e.update(state='running',phase='raster_contribution_ownership',peel=peel+1,mode=mode,frame=i+1,elapsed_seconds=time.time()-start)
            ratio=numerator/denominator.clamp_min(1e-10)
            accepted=(numerator>1e-4)&(ratio>.90)&~removed
            removed|=accepted
            row={'peel':peel+1,'new_points':int(accepted.sum()),'removed_total':int(removed.sum()),
                'mass_threshold':1e-4,'global_visible_mass_fraction_on_confirmed_road_or_selected_body_min':.90,
                'fraction_median_selected':float(ratio[accepted].median()) if accepted.any() else None}
            rounds.append(row);write(e.out/'rounds.json',rounds);print(json.dumps(row),flush=True)
            np.savez_compressed(e.out/('peel_%d_contribution.npz'%(peel+1)),numerator=numerator.cpu().numpy(),denominator=denominator.cpu().numpy())
        torch.save({'ground':removed.cpu(),'rounds':rounds,'source_checkpoint':str(args.restore),
            'freeze_static_in_road_pass':True,'road_authority_is_photometric_plus_LiDAR_and_far_actor_observation':True},e.out/'selection.pt')
        # Preview altered ownership and static-time removal before refitting.
        e.surface.ground.copy_(removed);e.surface.freeze_static.fill_(True);renderer.prepare_static=e.original_prepare
        with torch.no_grad():
            metrics=e.metrics([0,6,12,18,24,36,48]);e.preview('frozen_contribution_zero_update')
        write(e.out/'zero_update_metrics.json',metrics);e.verify_sources()
        e.update(state='complete',phase='actual_raster_ownership_ready',training_updates=0,rounds=rounds,zero_update_metrics=metrics)
    finally:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
if __name__=='__main__':run()
