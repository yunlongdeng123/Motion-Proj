"""Newly fitted ownership variant; never reuses mismatched weights as a result."""
import argparse,json,traceback
from train_surface import SurfaceExperiment,NEW
from ablate_ownership import ground_mask
from run_experiments import write
import renderer
class Repair(SurfaceExperiment):
    def __init__(self,args):
        super().__init__(args)
        if args.ownership in ['contribution_frozen','unselected_clean']:
            import torch
            folder='unselected_foreground_audit' if args.ownership=='unselected_clean' else 'contribution_guarded_audit'
            selection=torch.load(NEW/folder/'selection.pt',map_location='cpu',weights_only=False)
            mask=selection['ground'].to('cuda');self.surface.freeze_static.fill_(True)
        else:mask=ground_mask(self,args.ownership)
        self.surface.ground.copy_(mask)
        self.contract.update(ownership_refinement=args.ownership,
            ownership_semantics='Remove original below-plane representation and selected residual dark contributors only inside world-road routing authority; no semantic GT assertion',
            refitted_after_ownership_change=True,warm_start=str(args.restore),
            original_ground_removed='original strict set OR anchor height<.20m; optional diagnostic dark contributor IDs',
            color_space='parent rasterizer apparent sRGB alpha; shadow uses decode/encode of visible road contribution, approximate radiance transport')
        if args.ownership in ['contribution_frozen','unselected_clean']:
            self.contract.update(freeze_static_in_road_pass=True,ownership_method='true raster alpha*T feature gradients across49 views and original/frozen time; >90% global visible mass on authority road/selected body; three visibility peeling rounds',
                original_ground_removed='geometric below-plane base plus verified contribution_audit selection; former union extras re-evaluated; scene outside route unchanged')
            self.check_actor_responsibility()
        if args.ownership=='unselected_clean':self.contract.update(unselected_foreground_separated_from_frozen_road_pass=True,
            unselected_foreground_method='>70% actual visible mass on unselected semantic foreground OR >80% active-lifetime 3D-box membership, original time across49 frames; preserve bodies via original foreground pass')
        write(self.out/'runtime_contract.json',self.contract)
        write(self.out/'ownership_receipt.json',{'variant':args.ownership,'removed_parent_points':int(mask.sum()),'new_training_updates_planned':args.steps,'source_parameters_preserved':True})
    def check_actor_responsibility(self):
        import torch
        rows=[]
        with torch.no_grad():
            for i in [0,6,12,18,24,36,48]:
                new=self.render(i,shaded=True);body=self.body[i].any(0)
                renderer.prepare_static=self.original_prepare;renderer.prepare_instances=self.original_instances_prepare
                old=renderer.render(self.cameras[i],self.static,self.instances,self.config,self.bg,env_map=self.env,max_features=10,render_separate=False,is_training=False)
                renderer.prepare_instances=self.prepare_actor
                new_error=float(abs(new['render'][:,body]-self.gt[i,:,body]).mean());old_error=float(abs(old['render'][:,body]-self.gt[i,:,body]).mean())
                from bc_model import sample_render
                sm=self.surface_data['shadow'][i]&(self.surface_data['confidence']>.9)
                alpha=float(sample_render(new['dynamic_visible_alpha'],self.surface_data['uv'][i,sm]).mean()) if sm.any() else None
                rows.append({'frame':self.frame_ids[i],'original_body_RGB_MAE':old_error,'reassigned_body_RGB_MAE':new_error,'Actor_alpha_on_bodyfree_shadow_pseudo':alpha})
        regression=sum(r['reassigned_body_RGB_MAE']-r['original_body_RGB_MAE'] for r in rows)/len(rows)
        leaks=[r['Actor_alpha_on_bodyfree_shadow_pseudo'] for r in rows if r['Actor_alpha_on_bodyfree_shadow_pseudo'] is not None]
        mean_leak=sum(leaks)/len(leaks) if leaks else 0
        self.actor_opacity_enabled=regression>.005 or mean_leak>.02
        self.actor_opacity_delta.requires_grad_(self.actor_opacity_enabled)
        self.contract.update(actor_opacity_adapter_enabled=self.actor_opacity_enabled,
            actor_adapter_reason='Enable bounded opacity residual only if reassignment causes >.005 body RGB regression or measured alpha>.02 on body-free road-shadow samples; freeze geometry/SH and enforce independent body-mask/road-opacity losses',
            actor_opacity_odds_range='exp(3*tanh(delta)); supervised by full-scene per-ID visibility and body RGB, anchored to initial opacity')
        write(self.out/'actor_responsibility_check.json',{'per_frame':rows,'mean_body_RGB_regression':regression,'mean_Actor_alpha_on_bodyfree_shadow_pseudo':mean_leak,'opacity_adapter_enabled':self.actor_opacity_enabled,
            'geometry_and_SH_appearance_frozen':True,'mask_known_pairs':int(self.mask_known.sum()),'known_pairs_total':self.mask_known.numel()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--ownership',choices=['below','below_local','below_union','contribution_frozen','unselected_clean'],required=True);a=p.parse_args()
    warm='contribution_frozen_v1' if a.ownership=='unselected_clean' else 'repair_union_v1' if a.ownership=='contribution_frozen' else 'joint_v1'
    a.restore=str(NEW/warm/'final.pt');a.steps=2000 if a.ownership=='unselected_clean' else 4000;a.warmup=1;a.probe=False;a.plan_override=[('background',500),('shadow',250),('joint',1250)] if a.ownership=='unselected_clean' else [('background',1000),('shadow',500),('joint',2500)];e=None
    if a.ownership in ['contribution_frozen','unselected_clean']:a.guard_overrides=str(NEW/'guard_refinement_v2/overrides.npz')
    try:e=Repair(a);e.train()
    except BaseException as error:
        if e:e.update(state='failed',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if e:renderer.prepare_static=e.original_prepare;renderer.prepare_instances=e.original_instances_prepare;e.slot.close()
