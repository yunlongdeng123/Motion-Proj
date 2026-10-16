"""Actual full49 final/baseline rasterization, continuous editing and invariants."""
import sys,math,copy,json,time,argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from train_surface import SurfaceExperiment,ROOT,OLD,NEW
from run_experiments import write,save_image
from bc_model import ActorShadow,edited_casters,srgb_to_linear,linear_to_srgb,sample_render
from run_plane_refinement import PlaneRoad
from sequence_diagnostic import encode
import renderer

class SequenceEvaluator(SurfaceExperiment):
    def legacy_init(self):
        with np.load(OLD/'selection_plane_v1/selection.npz') as f:selection={k:f[k] for k in f.files}
        warm=torch.load(OLD/'runs/deferred_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
        s=ActorShadow(self.data['plane_model'].cpu().float(),self.data['world_scale']).cuda();s.load_state_dict(warm['shadow_state'])
        self.legacy=PlaneRoad(selection,self.static,self.instances,self.data,'R4',s).cuda();self.legacy.original_prepare=self.original_prepare
        self.legacy.load_addon_state(warm['road_addon'])
    @torch.no_grad()
    def legacy_render(self,i,instances=True,shaded=True,edits=None):
        a=self.legacy;a.current_frame=i;a.shaded=False;a.edits=edits
        renderer.prepare_static=a.patch
        renderer.prepare_instances=self.original_instances_prepare
        feature=torch.zeros(len(self.static.get_xyz),4,device='cuda')
        feature[a.receiver_ids]=torch.cat([torch.ones(len(a.receiver_ids),1,device='cuda'),a.color_raw.sigmoid()],1)
        pkg=renderer.render(self.cameras[i],self.static,self.instances if instances else None,self.config,self.bg,env_map=self.env,
            other=feature,edits=edits if instances else None,max_features=10,render_separate=False,is_training=False)
        renderer.prepare_static=self.original_prepare
        renderer.prepare_instances=self.prepare_actor
        if shaded and instances:
            q,_=self.road_pixels(self.cameras[i]);cast=edited_casters(self.instances,self.cameras[i].timestamp,edits,self.data['sizes_model'][i])
            att=a.shadow.attenuation(q,*cast).T.reshape(3,320,480)
            alpha=pkg['feature'][:1];road=pkg['feature'][1:4]
            repl=alpha*linear_to_srgb(srgb_to_linear((road/alpha.clamp_min(1e-7)).clamp(0,1))*att)
            pkg['render']=pkg['render']-road+torch.where(alpha>1e-7,repl,road)
        return pkg
    def point_metrics(self,i,clean,seen):
        d=self.surface_data;ids=torch.nonzero((d['confidence']>.9)&d['free'][i]).ravel()
        p=srgb_to_linear(sample_render(clean['render'],d['uv'][i,ids]));j=srgb_to_linear(sample_render(seen['render'],d['uv'][i,ids]));sm=d['shadow'][i,ids];cm=d['clean'][i,ids]
        p0=srgb_to_linear(sample_render(clean['render'],self.data['uv'][i]));j0=srgb_to_linear(sample_render(seen['render'],self.data['uv'][i]));a=self.data['accepted_reference'];s0=self.data['shadow'][i]
        uv0=self.data['uv'][i];view=(uv0[:,0]>=0)&(uv0[:,0]<=478)&(uv0[:,1]>=0)&(uv0[:,1]<=318)
        guard=sample_render(self.surface_data['protection'][i].float()[None],uv0)[:,0]>.01
        body=sample_render(F_dilate(self.body[i].any(0),3).float()[None],uv0)[:,0]>.01
        visible=a&view&~guard&~body
        if not hasattr(self,'footprint_points'):
            fp=np.asarray(Image.open(ROOT/'black_shadow_audit_20261016_v1/focus_cpu/reviewed_artifact_footprint.png'))>0
            pix=self.pixels.astype(int);self.footprint_points=torch.tensor(fp[pix[:,1],pix[:,0]],device='cuda')
        footprint=visible&self.footprint_points
        error=abs(p-d['reference'][ids]).mean(1)
        luma=torch.tensor([.2126,.7152,.0722],device='cuda');dark=(d['reference'][ids]-p)@luma
        actor_alpha=sample_render(seen['dynamic_visible_alpha'],d['uv'][i,ids])[:,0]
        return {'surface_clean_MAE':float(error.mean()),
            'observed_body_RGB_MAE':float(abs(seen['render'][:,self.body[i].any(0)]-self.gt[i,:,self.body[i].any(0)]).mean()),
            'observed_body_visible_ID_L1':float(((seen['instance_visible_contribution']-self.body[i]).abs()*self.mask_known[i,:,None,None]).sum()/(self.body[i]*self.mask_known[i,:,None,None]).sum().clamp_min(1)),
            'surface_clean_per_point_error_p99':float(torch.quantile(error,.99)),
            'surface_dark_residual_over006_fraction':float((dark>.06).float().mean()),
            'surface_bright_residual_over006_fraction':float((dark<-.06).float().mean()),
            'surface_shadow_MAE':float(abs(j[sm]-d['observed'][i,ids][sm]).mean()) if sm.any() else None,
            'Actor_visible_alpha_on_shadow_pseudo':float(actor_alpha[sm].mean()) if sm.any() else None,
            'surface_observed_clean_MAE':float(abs(j[cm]-d['observed'][i,ids][cm]).mean()) if cm.any() else None,
            'surface_clean_samples':len(ids),'surface_shadow_samples':int(sm.sum()),
            'old_local_clean_MAE':float(abs(p0[a]-self.data['reference_linear'][a]).mean()),
            'old_local_visible_clean_MAE':float(abs(p0[visible]-self.data['reference_linear'][visible]).mean()) if visible.any() else None,
            'old_local_visible_road_samples':int(visible.sum()),'old_local_reference_samples':int(a.sum()),
            'diagnostic_footprint_visible_clean_MAE':float(abs(p0[footprint]-self.data['reference_linear'][footprint]).mean()) if footprint.any() else None,
            'diagnostic_footprint_visible_samples':int(footprint.sum()),
            'old_local_shadow_MAE':float(abs(j0[s0]-self.data['observed_linear'][i,s0]).mean()) if s0.any() else None}

    @torch.no_grad()
    def run_evaluation(self):
        self.legacy_init();start=time.time()
        names=['observed','static','shadow_off','delete_target','delete_all','translate_smooth','yaw_smooth_15','yaw_smooth_30','fixed_static','fixed_translate','confidence','shadow_effect','comparison']
        for name in names:(self.out/name).mkdir()
        n=self.surface_data['plane'][:3];n=n/n.norm();tangent=self.surface_data['basis'][:,0]
        rows=[];fixed_ref=None;last_fixed=None;fixed_pair=[]
        for i in range(49):
            self.priority();fid=self.frame_ids[i];theta=2*math.pi*i/48
            move={0:{'translation':(tangent*math.sin(theta)*self.data['world_scale']).tolist()}}
            def yaw(deg):
                a=math.radians(deg)*math.sin(theta)
                return {0:{'quaternion':[math.cos(a/2),*(n*math.sin(a/2)).tolist()]}}
            seen=self.render(i,shaded=True);clean=self.render(i,instances=False);off=self.render(i)
            gone=self.render(i,shaded=True,edits={0:{'delete':True}})
            allgone=self.render(i,shaded=True,edits={k:{'delete':True} for k in range(self.instances.num_instances)})
            translated=self.render(i,shaded=True,edits=move);rotated=self.render(i,shaded=True,edits=yaw(15));rotated30=self.render(i,shaded=True,edits=yaw(30))
            cam=copy.copy(self.cameras[12]);cam.timestamp=self.cameras[i].timestamp
            fixed=self.render(i,instances=False,camera=cam);fixedmove=self.render(i,shaded=True,edits=move,camera=cam)
            legacyseen=self.legacy_render(i);legacyclean=self.legacy_render(i,instances=False,shaded=False)
            legacygone=self.legacy_render(i,edits={0:{'delete':True}})
            renderer.prepare_static=self.original_prepare
            renderer.prepare_instances=self.original_instances_prepare
            original=renderer.render(self.cameras[i],self.static,self.instances,self.config,self.bg,env_map=self.env,max_features=10,render_separate=False,is_training=False)
            originalclean=renderer.render(self.cameras[i],self.static,None,self.config,self.bg,env_map=self.env,max_features=10,render_separate=False,is_training=False)
            originalgone=renderer.render(self.cameras[i],self.static,self.instances,self.config,self.bg,env_map=self.env,edits={0:{'delete':True}},max_features=10,render_separate=False,is_training=False)
            renderer.prepare_instances=self.prepare_actor
            field=seen['road_confidence'];confidence=torch.cat([field*.1,field,field*.1],0)
            effect=(off['render']-seen['render']).abs().mean(0,keepdim=True).clamp(0,1)
            effect=torch.cat([effect*5,effect*.8,effect*.2],0).clamp(0,1)
            packs=[seen,clean,off,gone,allgone,translated,rotated,rotated30,fixed,fixedmove,{'render':confidence},{'render':effect}]
            for name,pkg in zip(names,packs):save_image(self.out/name/(fid+'.png'),pkg['render'])
            sheet=Image.new('RGB',(960,688),'white');draw=ImageDraw.Draw(sheet)
            for k,(label,pkg) in enumerate([('original checkpoint',original),('retained deferred_R4',legacyseen),('joint observed',seen),('joint delete target',gone)]):
                x=k%2*480;y=k//2*344
                im=Image.fromarray(np.rint(pkg['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8'))
                sheet.paste(im,(x,y+24));draw.text((x+6,y+5),fid+' '+label,fill='black')
            sheet.save(self.out/'comparison'/(fid+'.png'))
            support=field[0]==0;foreground=F_dilate(self.body[i].any(0),3)
            scenery=support&~foreground
            fixedinterior=fixed['road_confidence'][0]>.99999
            if fixed_ref is None:fixed_ref=fixed['render'].clone();refmask=fixedinterior.clone()
            common=fixedinterior&refmask
            fixedchange=float(abs(fixed['render'][:,common]-fixed_ref[:,common]).mean()) if common.any() else None
            if last_fixed is not None:fixed_pair.append(float(abs(fixed['render']-last_fixed).mean()))
            last_fixed=fixed['render'].clone()
            row={'frame':fid,'final':self.point_metrics(i,clean,seen),'deferred':self.point_metrics(i,legacyclean,legacyseen),'original':self.point_metrics(i,originalclean,original),
                'delete_all_identity_max':float(abs(allgone['render']-clean['render']).max()),
                'outside_authorized_road_scenery_max_vs_parent':float(abs(seen['render'][:,scenery]-original['render'][:,scenery]).max()),
                'outside_authorized_road_scenery_MAE_vs_parent':float(abs(seen['render'][:,scenery]-original['render'][:,scenery]).mean()),
                'road_routing_outside_max_with_same_actor_gates':float(abs(seen['render'][:,support]-seen['parent_baseline_with_same_actor_gates'][:,support]).max()),
                'road_confident_pixels':int((field[0]>.99999).sum()),
                'fixed_supported_road_MAE_vs_first':fixedchange,'fixed_supported_road_pixels_common':int(common.sum()),
                'fixed_full_image_MAE_vs_first':float(abs(fixed['render']-fixed_ref).mean()),
                'observed_vs_GT_sRGB_MAE':float(abs(seen['render']-self.gt[i]).mean()),
                'shadow_effect_nonzero_pixels':int(((off['render']-seen['render']).abs().max(0).values>1e-5).sum()),
                'shadow_on_opaque_actor_max':float(abs(off['render']-seen['render'])[:,seen['dynamic_visible_alpha'][0]>.999].max()) if (seen['dynamic_visible_alpha'][0]>.999).any() else 0.}
            q=self.surface_data['xyz'];p=self.data['poses'][i].float();size=self.data['sizes_model'][i].float()
            local=torch.einsum('kni,kij->knj',q[None]-p[:,:3,3,None].transpose(1,2),p[:,:3,:3])/self.data['world_scale']
            distance=(abs(local[:,:,:2])-size[:,None,:2]/self.data['world_scale']/2).clamp_min(0).norm(dim=-1)
            distance[~self.data['pose_valid'][i]]=float('inf')
            assoc=self.surface_data['shadow'][i]&(self.surface_data['confidence']>.9)&(distance[0]+.5<distance[1:].min(0).values)
            query=self.surface_data['uv'][i,assoc];reference=self.surface_data['reference'][assoc]
            row['deleted_target_associated_shadow_pseudo_samples']=int(assoc.sum())
            for variant,pkg in [('original',originalgone),('deferred',legacygone),('final',gone)]:
                row[variant]['deleted_target_shadow_area_clean_MAE']=float(abs(srgb_to_linear(sample_render(pkg['render'],query))-reference).mean()) if len(query) else None
            rows.append(row);write(self.out/'per_frame.json',rows)
            self.update(state='running',phase='final49_continuous_evaluation',frame=i+1,elapsed_seconds=time.time()-start)
            print(json.dumps(row),flush=True)
        videos=[encode(self.out/name) for name in names]
        summaries={}
        for variant in ['original','deferred','final']:
            summaries[variant]={key:float(np.mean([r[variant][key] for r in rows if r[variant][key] is not None])) for key in ['surface_clean_MAE','surface_shadow_MAE','surface_observed_clean_MAE','observed_body_RGB_MAE','observed_body_visible_ID_L1','old_local_clean_MAE','old_local_visible_clean_MAE','old_local_shadow_MAE','diagnostic_footprint_visible_clean_MAE','deleted_target_shadow_area_clean_MAE']}
        weighted={v:{key:float(sum(r[v][key]*r[v][count] for r in rows if r[v][key] is not None)/sum(r[v][count] for r in rows if r[v][key] is not None)) for key,count in [('surface_clean_MAE','surface_clean_samples'),('surface_shadow_MAE','surface_shadow_samples'),('diagnostic_footprint_visible_clean_MAE','diagnostic_footprint_visible_samples')]} for v in ['original','deferred','final']}
        summary={'frames':49,'video_streams':len(names),'elapsed_seconds':time.time()-start,'metrics_mean_frames':summaries,'metrics_pixel_weighted':weighted,
            'max_delete_identity':max(r['delete_all_identity_max'] for r in rows),
            'max_scenery_change_outside_road':max(r['outside_authorized_road_scenery_max_vs_parent'] for r in rows),
            'max_road_routing_outside_same_actor_gates':max(r['road_routing_outside_max_with_same_actor_gates'] for r in rows),
            'mean_fixed_supported_road_time_MAE':float(np.mean([r['fixed_supported_road_MAE_vs_first'] for r in rows if r['fixed_supported_road_MAE_vs_first'] is not None])),
            'mean_fixed_full_image_time_MAE':float(np.mean([r['fixed_full_image_MAE_vs_first'] for r in rows])),
            'fixed_full_consecutive_mean':float(np.mean(fixed_pair)),
            'shadow_parameters':self.surface.shadow.describe(),'actor_gates':self.actor_gate_raw.sigmoid().tolist(),
            'all_parent_and_actor_source_parameters_preserved':True,'videos':videos,
            'scope':'49 same reconstruction training frames; confidence-supported road only; no paired edit GT or unseen generalization claim'}
        self.verify_sources();write(self.out/'summary.json',summary)
        self.update(state='complete',phase='final49_evaluation_complete',summary=summary)

def F_dilate(mask,n):
    return torch.nn.functional.max_pool2d(mask.float()[None,None],2*n+1,1,n)[0,0].bool()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--restore',required=True);a=p.parse_args();a.steps=0;a.warmup=1;a.probe=True;e=None
    try:e=SequenceEvaluator(a);e.run_evaluation()
    finally:
        if e:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
