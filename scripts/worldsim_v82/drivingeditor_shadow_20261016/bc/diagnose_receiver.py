"""Read-only actual-checkpoint test of the shadow receiver coverage and RGB floor."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from run_experiments import Experiment,BASE,write,save_image
from bc_model import ActorShadow,srgb_to_linear,sample_render
import renderer

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();a.steps=1;a.warmup=1
e=Experiment(a)
try:
    addon=torch.load(BASE/'runs/pilot_1000_v1/R4/addon_001000.pt',map_location='cpu',weights_only=False)
    shadow=ActorShadow(e.data['plane_model'].cpu(),e.data['world_scale'],'projection').cuda();shadow.load_state_dict(addon['shadow_state'])
    e.activate('R4',shadow);e.active.load_addon_state(addon['road_addon'])
    with torch.no_grad():
        original=e.render(12,shaded=True)
        e.active.shaded=False;e.active.current_frame=12
        feature=torch.zeros((len(e.static.get_xyz),1),device='cuda');feature[e.active.receiver_ids]=1
        pkg=renderer.render(e.cameras[12],e.static,e.instances,e.config,e.bg,env_map=e.env,other=feature,render_separate=False,max_features=10)
        fraction=sample_render(pkg['feature'],e.data['uv'][12])[:,0]
        sm=e.data['shadow'][12];target=e.data['observed_linear'][12]
        # Full black receivers give a strict lower image bound for this receiver set.
        native=shadow.attenuation
        shadow.attenuation=lambda receiver,*args:torch.zeros((len(receiver),3),device=receiver.device)
        black=e.render(12,shaded=True);shadow.attenuation=native
        lower=srgb_to_linear(sample_render(black['render'],e.data['uv'][12]))
        observed=srgb_to_linear(sample_render(original['render'],e.data['uv'][12]))
        result={'schema':'shadow_receiver_capacity_diagnosis_v1','training_updates':0,
            'actual_R4_shadow_linear_MAE':float(abs(observed[sm]-target[sm]).mean()),
            'unavoidable_positive_linear_error_even_with_black_receivers':float((lower[sm]-target[sm]).clamp_min(0).mean()),
            'receiver_visible_alpha_T_mean_in_shadow_region':float(fraction[sm].mean()),
            'fraction_shadow_channels_where_unshaded_others_exceed_GT':float((lower[sm]>target[sm]+.005).float().mean()),
            'all_black_receiver_image_is_not_a_repair':True}
        save_image(e.out/'R4_original.png',original['render']);save_image(e.out/'receivers_black_lower_bound_NOT_A_REPAIR.png',black['render'])
        write(e.out/'diagnosis.json',result);e.verify_sources();e.update(state='complete',phase='read_only_receiver_diagnosis',result=result)
        print(json.dumps(result))
finally:
    renderer.prepare_static=e.original_prepare;e.slot.close()
