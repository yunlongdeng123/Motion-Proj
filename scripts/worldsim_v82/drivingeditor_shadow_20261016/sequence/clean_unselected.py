"""Separate unselected vehicle foreground from the time-frozen road pass."""
import argparse,json
import numpy as np
import torch
import torch.nn.functional as F
from train_surface import SurfaceExperiment,NEW
from run_experiments import write
import renderer
def run():
    a=argparse.Namespace(output=str(NEW/'unselected_foreground_audit'),steps=0,warmup=1,restore=str(NEW/'contribution_frozen_v1/final.pt'),probe=True)
    e=SurfaceExperiment(a)
    try:
        removed=e.surface.ground.clone();n=len(removed);feature=torch.zeros(n,2,device='cuda',requires_grad=True)
        numerator=torch.zeros(n,device='cuda');denominator=torch.zeros(n,device='cuda')
        geometric_hit=torch.zeros(n,device='cuda');active_weight=torch.zeros(n,device='cuda')
        for i,cam in enumerate(e.cameras):
            if i%7==0:e.priority()
            with torch.no_grad():
                xyz=e.static.get_xyz_SHM(cam.timestamp);m=e.static.get_marginal_t(cam.timestamp).ravel()
                active=(m>.05)&~removed&(e.static.get_opacity.ravel()>.001)
                weight=m*active.float();active_weight+=weight;hit=torch.zeros(n,device='cuda',dtype=torch.bool)
                eligible=active&((xyz@e.surface.plane[:3]+e.surface.plane[3])/e.surface.scale>.15)
                for corners in e.protection_boxes[i]:
                    c=torch.tensor(corners,device='cuda',dtype=torch.float32);center=c.mean(0)
                    edge=torch.stack([c[4]-c[0],c[2]-c[0],c[1]-c[0]],1);size=edge.norm(dim=0);rot=edge/size[None]
                    extent=(rot.abs()@(size/2+.005))
                    ids=torch.nonzero(eligible&((xyz-center).abs()<extent).all(1)).ravel()
                    if len(ids):
                        local=(xyz[ids]-center)@rot;inside=(abs(local)<size/2+.005).all(1);hit[ids[inside]]=True
                geometric_hit+=weight*hit.float()
            renderer.prepare_static=e.original_prepare;feature.grad=None
            pkg=renderer.render(cam,e.static,None,e.config,e.bg,env_map=e.env,other=feature,mask=~removed,max_features=10,render_separate=False,is_training=False)
            mask=e.surface_data['protection'][i].float()
            loss=(pkg['feature'][0]*mask).sum()+pkg['feature'][1].sum();loss.backward()
            assert torch.isfinite(feature.grad).all();numerator+=feature.grad[:,0];denominator+=feature.grad[:,1]
            e.update(state='running',phase='unselected_foreground_ownership',frame=i+1)
        mass_fraction=numerator/denominator.clamp_min(1e-10);geo_fraction=geometric_hit/active_weight.clamp_min(1e-10)
        by_mass=(numerator>1e-4)&(mass_fraction>.70)&~removed
        by_geometry=(geo_fraction>.80)&(active_weight>.05)&~removed
        foreground=by_mass|by_geometry;removed|=foreground
        torch.save({'ground':removed.cpu(),'foreground_ids':torch.nonzero(foreground).ravel().cpu(),
            'source_checkpoint':a.restore,'mass_threshold':.70,'geometric_lifetime_fraction':.80},e.out/'selection.pt')
        receipt={'foreground_points':int(foreground.sum()),'by_visible_mass':int(by_mass.sum()),'by_3D_box_lifetime':int(by_geometry.sum()),
            'removed_total':int(removed.sum()),'unselected_bodies_still_retained_via_semantic_guard_and_original_full_scene_pass':True,
            'road_pass_only':True,'meaning':'foreground separation; not deleting unselected vehicles from final scene'}
        write(e.out/'receipt.json',receipt)
        e.surface.ground.copy_(removed)
        with torch.no_grad():e.preview('zero_update_foreground_separation');metrics=e.metrics([0,6,12,18,24,36,48])
        write(e.out/'zero_update_metrics.json',metrics);e.verify_sources()
        e.update(state='complete',phase='unselected_foreground_selection_ready',training_updates=0,receipt=receipt)
    finally:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
if __name__=='__main__':run()
