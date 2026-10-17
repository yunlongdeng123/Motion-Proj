"""Full49 actual shaded dynamic/edit streams and fixed-state shadow checks."""
from pathlib import Path
import sys,json,time
import torch,numpy as np
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'backend_restore_r15';sys.path.insert(0,str(R/'implementation_r15_train_v3/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.model import SceneModel
from shadow_rebuild.data import Dataset
from shadow_rebuild.train import seed_from_state,write
from shadow_rebuild.multi_evaluate import evaluate_many,edit_rotation
torch.set_num_threads(2);saved=torch.load(D/'main/latest.pt',map_location='cpu',weights_only=False)
if saved['global_step']!=45000 or saved['stage']!='joint':raise ValueError('full planned training has not finished')
raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate();data=Dataset(D/'idsplat_frontend/data_IDSplat','cuda',True);model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.requires_grad_(False)
if saved['input_contract']!=data.manifest:raise ValueError('trained input differs')
targets=[5,6,12,18,25,26,37];start=time.monotonic();protocol=json.loads((R/'comparison_protocol.json').read_text());records=evaluate_many(model,data,D/'final49',targets,protocol=protocol)
tests=[]
with torch.no_grad():
    for actor,index in [(5,18),(6,16),(12,48),(18,48),(25,48),(26,44),(37,32)]:
        f=data.frame(index);g,p=model.buffers(f,len(data.actor_ids),'cuda',isolated=False)
        keep=(p['road_alpha']>.05)&(p['road_normal'].norm(dim=-1)>.5);ids=keep.flatten().nonzero().flatten()
        if not len(ids):tests.append(dict(actor=actor,status='NO_VISIBLE_ROAD_RECEIVERS'));continue
        position=p['road_position'].reshape(-1,3)[ids];normal=p['road_normal'].reshape(-1,3)[ids];albedo=p['road_albedo'].reshape(-1,3)[ids]
        car=g['xyz'][g['owner']==actor]
        if not len(car):tests.append(dict(actor=actor,status='ACTOR_NOT_PRESENT'));continue
        center=car.median(0).values;light=model.light.direction();height=(center[2]-position[:,2].median()).clamp_min(.1);foot=center-light*height/light[2].clamp_min(.1)
        choice=((position[:,:2]-foot[:2]).square().sum(-1)).topk(min(2048,len(ids)),largest=False).indices;position,normal,albedo=position[choice],normal[choice],albedo[choice];receiver=torch.ones(len(choice),device='cuda')
        c0,rho,vis=model.light.unshadowed_and_rho(position,normal,albedo,g,receiver)
        results={}
        for name,edit in [('delete',{'delete':True}),('translate',{'translation':[1.,0.,0.]}),('yaw15',{'yaw':float(np.deg2rad(15))}),('yaw30',{'yaw':float(np.deg2rad(30))})]:
            eg=model.field.evaluate(f.poses,f.valid,f.camera.center,{actor:edit});base,r,v=model.light.unshadowed_and_rho(position,normal,albedo,eg,receiver)
            results[name]=dict(rho_mean_abs_change=float((r-rho).abs().mean()),base_color_max_change=float((base-c0).abs().max()),remaining_actor_Gaussians=int((eg['owner']==actor).sum()))
        same=model.field.evaluate(f.poses.clone(),f.valid.clone(),f.camera.center);_,fixed,_=model.light.unshadowed_and_rho(position,normal,albedo,same,receiver)
        _,off,_=model.light.unshadowed_and_rho(position,normal,albedo,g,receiver,enabled=False)
        only={key:value[g['owner']==actor] for key,value in g.items()};_,solo,_=model.light.unshadowed_and_rho(position,normal,albedo,only,receiver)
        empty={key:value[:0] for key,value in g.items()};_,removed,_=model.light.unshadowed_and_rho(position,normal,albedo,empty,receiver)
        tests.append(dict(actor=actor,raw=index+89,receivers=len(choice),baseline_rho_mean=float(rho.mean()),solo_actor_rho_max=float(solo.max()),delete_solo_rho_max=float(removed.max()),shadow_off_rho_max=float(off.max()),fixed_pose_camera_light_repeat_max=float((fixed-rho).abs().max()),pose_edits_at_same_frame=results,
            interpretation='fixed receiver/material diagnostic isolates geometry/pose-driven attenuation; actual edited visibility appears in49frame streams; no edited/no-shadow GT'))
write(D/'shadow_pose_checks.json',dict(tests=tests,no_time_embedding=True,GT_trajectory_not_replaced=True,base_shadow_geometry_shared=True,actual_full49_done=True,seconds=time.monotonic()-start))
write(D/'full_experiment_complete.json',dict(training_updates=45000,geometry=30000,material=10000,joint=5000,full49=True,streams=65,root_visual_review_pending=True,auto_shutdown=False,problem_solved=None,human_verdict=None))
print('R14_FULL_TRAINING_AND49FRAME_EVALUATION_COMPLETE',flush=True)
