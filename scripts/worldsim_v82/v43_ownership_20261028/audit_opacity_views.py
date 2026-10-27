"""Common-domain coverage, boundary leakage and native edits after formal arms finish."""
import json,time,argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from train_actor import Actor,NativeMotion,ROOT,OLD,torch,cpu,render
from gpu_slot import exclusive
from v43_resource_control import require_start
from diagnose_first_badcases import save
from opacity_calibration import attach_saved_field
from segmentation import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--boundary-only',action='store_true');args=p.parse_args();ctx='14705303724557273004_3105_000_3125_000';require_start(ctx);R=ROOT/'first_gpu_review_v1';out=R/('boundary_view_audit' if args.boundary_only else 'opacity_view_audit');out.mkdir(exist_ok=False)
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);assert json.loads((R/'opacity_controller.json').read_text())['phase']=='complete';ds=ROOT/'datasets'/ctx;frames=json.loads((ds/'manifest.json').read_text())['frames'];n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);actor=Actor(np.load(ds/'seeds.npz'));base=torch.load(str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor'];actor.load_state_dict(base);variants={}
        for arm in (['balanced','boundary'] if args.boundary_only else ['legacy','balanced','field']):
            variants[arm]=np.load(R/'opacity'/arm/'formal/evaluation/010000/effective_opacity.npz')['opacity']
        o=cpu(base['opacity'].sigmoid()[:,0]);variants=({'base':o,**variants} if args.boundary_only else {'base':o,'fixed2':1-(1-o)**2,**variants});records=[];cases={c:json.loads((OLD/'motion_edit_suite_v1'/ctx/(c+'.json')).read_text())['frames'] for c in ['no_edit','turn_left15','translate_lateral_1m']}
        field_state=torch.load(str(R/'opacity/field/formal/checkpoint010000.pth'),map_location='cuda')['field'];actor.load_state_dict(base);generic=attach_saved_field(actor,field_state)
        expected_field=np.load(R/'opacity/field/formal/evaluation/010000/effective_opacity.npz')['opacity'];generic_opacity=-torch.expm1(torch.log1p(-actor.opacity.sigmoid().clamp(1e-6,1-1e-6))*generic.calibration.factors());module_diff=float((generic_opacity[:,0]-torch.from_numpy(expected_field).cuda()).abs().max());assert module_diff<1e-7
        for case,rows in cases.items():
            for row in rows:
                i=row['camera_index'];f=frames[i];raw=f['raw'];body=np.load(ds/('%03d.npz'%raw))['body'];outer=binary_dilation(body,iterations=5)&~body;edit=None if case=='no_edit' else dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']);actor.load_state_dict(base);baseline=render(n,actor,f,i,static=False,edit=edit);domain=baseline['alpha'][0]>.1
                for name,opacity in variants.items():
                    actor.opacity.copy_(torch.logit(torch.from_numpy(opacity).cuda().clamp(1e-6,1-1e-6))[:,None]);iso=render(n,actor,f,i,static=False,edit=edit);r=render(n,actor,f,i,edit=edit);alpha=iso['alpha'][0];weight=r['extra'][0];a=cpu(alpha);rec=dict(case=case,raw=raw,variant=name,common_domain='baseline same-pose isolated alpha>.1',domain_pixels=int(domain.sum()),isolated_alpha_on_common_domain=float(alpha[domain].mean()),opaque95_fraction_on_common_domain=float((alpha[domain]>.95).float().mean()),visible_mass_on_common_domain=float(weight[domain].sum()),visible_fraction_on_common_domain=float((weight[domain]/alpha[domain].clamp_min(1e-6)).clamp(0,1).mean()))
                    if case=='no_edit':rec.update(inside_mask_alpha=float(a[body].mean()),outside5px_alpha=float(a[outer].mean()),outside5px_alpha_above_point1_pixels=int((a[outer]>.1).sum()),whole_outside_alpha=float(a[~body].mean()),split='holdout' if i%3==1 else 'train')
                    records.append(rec)
                    if raw in [91,98,101,109,113,114]:save(out/case/name/('%03d.png'%raw),r['rgb']);save(out/case/name/('%03d_actor.png'%raw),iso['rgb'])
        actor.load_state_dict(base);assert n.unchanged();write(out/'summary.json',dict(records=records,generic_module_opacity_max_abs_difference=module_diff,training_updates=0,frozen_structure_and_DC=True,RGB_scores_used=False,no_edited_GT=True,quality_accepted=False,human_verdict=None));print('VIEW_AUDIT_COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
