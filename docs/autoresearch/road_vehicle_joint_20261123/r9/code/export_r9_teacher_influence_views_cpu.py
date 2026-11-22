"""Display-only per-view field-influence reads; no field/acceptance mutations."""
from pathlib import Path
import numpy as np
from audit_r9_shared_field_cpu import load,pin,write,gather,png

WORK=Path(__file__).resolve().parents[2];R9=WORK/'old_raw_teacher_joint_20261123_r9';R7=WORK/'road_shared_appearance_20261122_r7'
for variant,role in [('joint_shared_v1','Actual direct RGB blend coefficient; old v1 visibility feather'),
                     ('joint_shared_continuous_v2','Conditional influence of teacher data under the continuous PDE; NOT a pointwise RGB multiplier or calibrated probability')]:
    field_path=R9/variant/'final_atlas.npz';f=load(field_path);out=R9/'cpu_review'/variant;rows=[]
    for frame in [86,110,134,179,198]:
        cp=R7/f'projection_cache_v1/{frame:03d}/view_cache.npz';c=load(cp);mask_path=out/f'{frame:03d}_domain_masks.npz';m=load(mask_path)['evaluated']
        value,_,_=gather(f['effective_teacher_blend'],c['atlas_uv_map']);value=np.where(m,np.clip(value,0,1),0).astype(np.float32)
        target=out/f'{frame:03d}_teacher_alpha.png';assert not target.exists();png(target,value)
        np.savez_compressed(out/f'{frame:03d}_teacher_influence_display.npz',teacher_influence=value,material_display_valid=m)
        rows.append({'raw':frame,'PNG':pin(target),'float_display':pin(out/f'{frame:03d}_teacher_influence_display.npz'),
          'projection_cache':pin(cp),'display_domain':pin(mask_path),'material_pixels':int(m.sum()),
          'value_min':float(value[m].min()),'value_max':float(value[m].max()),'value_mean':float(value[m].mean())})
    write(out/'teacher_influence_display_receipt.json',{'schema':'R9.per_view_teacher_influence_display.v1','complete':True,
      'variant':variant,'field':pin(field_path),'field_array':'effective_teacher_blend','quantity_role':role,
      'readout':'Bilinear atlas UV from the original camera cache; exactly the independently frozen actual-material display mask. Zero outside display domain is a PNG sentinel.',
      'calibrated_probability':False,'field_or_prior_QA_modified':False,'rows':rows,'CPU_only':True,'GPU_forward':0,'diffusion_calls':0,'Gaussian_updates':0})
print('Complete: both variants, five original-resolution influence PNG/float reads each.')
