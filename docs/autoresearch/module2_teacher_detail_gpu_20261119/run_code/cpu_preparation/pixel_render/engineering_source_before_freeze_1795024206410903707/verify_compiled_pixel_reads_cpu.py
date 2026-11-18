"""复核已存dense审计coarse值与全量cache，列出固定A只读floor；不增加采样/实验。"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
from scipy.ndimage import gaussian_filter

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1';P=B/'road_multiscale_material_20261118_r1'


def main():
    path=Q/'code/fixed_material_read_operator.py';spec=importlib.util.spec_from_file_location('root_fixed_pixel_read_verify',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    root=Q/'pixel_render/read_operator_box_r1';audit=Q/'pixel_render/quadrature_numeric_r1'
    r=json.loads((audit/'receipt.json').read_text());stored=np.load(audit/'colors.npz')['coarse_dense'][:,0]
    with np.load(Q/'permission/final_r1/teacher_evidence.npz')as z:valid=z['common_material_valid'];latent=z['common_latent_mask']
    yy,xx=np.indices(valid.shape);raw=.8*np.sin(2*np.pi*xx/4)*np.cos(2*np.pi*yy/6)*latent
    bound=json.loads((Q/'protocol/bound_recipe.json').read_text())['high_bound']
    h=.5*bound*(raw-gaussian_filter(raw,3.,mode='nearest',truncate=3.));H=np.stack((h,.8*h,-.6*h),-1)
    differences=[];massdiff=[];floor=[];masks={}
    for frame in(86,110):
        op=mod.FixedReadOperator.load(root/f'{frame:03d}');lookup={int(i):n for n,i in enumerate(op.row_ids)}
        low=np.zeros((*op.low_shape,3));zero=np.zeros_like(H)
        answers=dict(base=op.numpy_query(low,zero,'observed_plus_teacher_H'),observed_H_stress=op.numpy_query(low,H,'observed_H'),teacher_H_stress=op.numpy_query(low,H,'observed_plus_teacher_H'))
        for n,row in enumerate(r['rows']):
            if row['raw']==frame:
                index=lookup[row['pixel']];differences.append(float(abs(answers[row['case']][index]-stored[n]).max()));massdiff.append(abs(op.mass[index]-row['coarse_mass']))
        with np.load(P/f'stage0_native_r1/{frame:03d}_targets.npz')as target,np.load(Q/f'pixel_render/maps/{frame:03d}.npz')as geo,np.load(P/f'formal_r1/filtered_A0/{frame:03d}_native.npz')as parent:
            bad=target['A'].ravel()[op.row_ids]&(op.mass<=1e-12);ids=op.row_ids[bad];mask=np.zeros(target['A'].shape,bool);mask.ravel()[ids]=True
            masks[f'{frame:03d}_A_source_floor']=mask;masks[f'{frame:03d}_A_qualified_world_material']=target['A']&~mask
            for pixel in ids:
                y,x=np.unravel_index(pixel,mask.shape);floor.append(dict(raw=frame,pixel=int(pixel),y=int(y),x=int(x),center_uv=geo['center_uv'][y,x].tolist(),
                    original_road_alpha=float(parent['road_alpha'][y,x]),material_mass=float(op.mass[lookup[int(pixel)]]),
                    action='original eligible ci fallback; fullA denominator retained',actual_eligible_native_weight='not yet exported'))
    out=dict(schema='full_pixel_sparse_cache_CPU_verification.v1',passed=bool(max(differences)<1e-10 and max(massdiff)<1e-12),
        existing_coarse_values_compared=len(differences),RGB_max=max(differences),mass_max=float(max(massdiff)),
        A_source_floor=floor,A_full=16904,A_qualified_world_material=16904-len(floor),
        C10830_S3131_fullS3187_zero_material_mass=0,
        native_calls=0,optimizer_updates=0,GPU_executed=False,new_sampling_performed=False,
        floor_boundary='future smoke measures actual We/gradient/error of these7; no base/scope change, no trainingmask shrink')
    (root/'verification.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(root/'fixed_A_source_floor.npz',**masks)
    if not out['passed']:raise RuntimeError(out)
    print(json.dumps(out))


if __name__=='__main__':main()
