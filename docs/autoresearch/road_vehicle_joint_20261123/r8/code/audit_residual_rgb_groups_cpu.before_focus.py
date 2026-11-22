"""Read actual additive RGB groups and propose image-local ground candidates."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation
from PIL import Image
from verify_joint_asset_cpu import read,write,pin,array_record


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def bbox(mask):
    y,x=np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1] if len(x) else None
def all4(a,uv):
    q=np.nan_to_num(uv);x=np.floor(np.clip(q[...,0],0,a.shape[1]-1)).astype(int);y=np.floor(np.clip(q[...,1],0,a.shape[0]-1)).astype(int)
    jx=np.minimum(x+1,a.shape[1]-1);jy=np.minimum(y+1,a.shape[0]-1)
    return a[y,x]&a[y,jx]&a[jy,x]&a[jy,jx]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    C=a.root.parent;source=a.root/'residual_rgb_groups_v1';receipt=read(source/'receipt.json');assert receipt['complete'] and receipt['actual_nativeF']==2
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    field=load(C/'old_raw_teacher_joint_20261123_r9/joint_shared_continuous_v2/final_atlas.npz')
    for frame in receipt['rows']:
        raw=frame['raw'];path=source/f'{raw:03d}/rgb_groups.npz';z=load(path);G=z['group_RGB_premult'];A=z['group_alphaT'];rgb=z['rgb_with_environment']
        expected_path=a.root/f'joint_rgb_formal400_v1/final_readout/{raw:03d}/joint_render.npz';expected=load(expected_path)
        assert array_record(rgb)==array_record(expected['rgb']) and array_record(z['native_alpha'])==array_record(expected['alpha'])
        assert G.shape==(5,3,768,1152) and A.shape==(5,768,1152) and np.isfinite(G).all() and np.isfinite(A).all()
        rgb_error=float(abs(G.astype(np.float64).sum(0)-z['rgb_native_premult']).max());alpha_error=float(abs(A.astype(np.float64).sum(0)-z['native_alpha']).max())
        assert rgb_error<=2e-5 and alpha_error<=2e-5
        x0,y0,x1,y1=frame['fixed_relative_lower_edge_ROI_xyxy'];roi=np.zeros(A.shape[1:],bool);roi[y0:y1,x0:x1]=True
        masks_path=a.root/f'audit/rectified_masks_v1/{raw:03d}/rectified_masks.npz';m=load(masks_path)
        component_path=a.root/f'joint_rgb_formal400_v1/final_readout/{raw:03d}/component_render.npz';component=load(component_path)
        body=binary_dilation((component['actor_alpha']>.05)|m['actor_visible'],iterations=3)
        ground=roi&~body&~m['protected']
        # Post-hoc diagnostic colour regions, not training labels or new QA domains.
        red=ground&(rgb[0]>.2)&(rgb[0]>np.maximum(rgb[1],rgb[2])+.12)
        blue=ground&(rgb[2]>rgb[0]+.03)&(rgb[2]>rgb[1]+.015)&(rgb.max(0)>.08)
        dark=ground&(rgb.max(0)<.22)&(A[3]>.25)
        spectral=red|blue|dark
        cache_path=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz';cache=load(cache_path)
        map_path=Path(str(cache['source_map_path']));source_map=load(map_path)
        surface_path=C/f'old_raw_teacher_joint_20261123_r9/audit/surface_visibility/{raw:03d}/surface.npz';surface=load(surface_path)
        physical=cache['geometry_valid']&cache['projection_valid']&surface['triangle_all_corners_geometry_valid']&np.isfinite(source_map['world_xyz']).all(-1)
        legal=all4(field['permission_mask']&(field['source_kind']>0),cache['atlas_uv_map'])
        summaries={}
        regions={'original_proxy_lower_edge_ROI':roi,'protected_ground_candidate':ground,'red_diagnostic':red,'blue_cyan_diagnostic':blue,'dark_context_diagnostic':dark,'union_diagnostic':spectral}
        for name,mask in regions.items():
            value=G[:,:,mask].astype(np.float64);alpha=A[:,mask].astype(np.float64);mass=np.abs(value).sum((1,2));den=max(mass.sum(),1e-30)
            good=mask&physical&legal;points=source_map['world_xyz'][good];uv=cache['atlas_uv_map'][good]
            summaries[name]={'pixels':int(mask.sum()),'bbox_xyxy':bbox(mask),'group_RGB_channel_sum':value.sum(2).tolist(),
              'group_RGB_absolute_mass_fraction':(mass/den).tolist(),'group_alphaT_sum':alpha.sum(1).tolist(),
              'group_alphaT_fraction':(alpha.sum(1)/max(alpha.sum(),1e-30)).tolist(),
              'canonical_isolated_body_excluded':name!='original_proxy_lower_edge_ROI',
              'reliable_existing_road_XYZ_and_legal_material_pixels':int(good.sum()),
              'existing_geometry_or_legal_material_unsupported_pixels':int((mask&~(physical&legal)).sum()),
              'reliable_road_world_XYZ_min':points.min(0).tolist() if len(points) else None,
              'reliable_road_world_XYZ_max':points.max(0).tolist() if len(points) else None,
              'reliable_road_atlas_UV_min':uv.min(0).tolist() if len(uv) else None,
              'reliable_road_atlas_UV_max':uv.max(0).tolist() if len(uv) else None}
        folder=a.output/f'{raw:03d}';folder.mkdir()
        payload={name:mask[y0:y1,x0:x1] for name,mask in regions.items()}
        np.savez_compressed(folder/'ROI_attribution.npz',group_RGB_premult=G[:,:,y0:y1,x0:x1],group_alphaT=A[:,y0:y1,x0:x1],
            actual_shared_RGB=rgb[:,y0:y1,x0:x1],canonical_body_protection=body[y0:y1,x0:x1],
            geometry_and_material_legal=(physical&legal)[y0:y1,x0:x1],**payload)
        for name,mask in regions.items():Image.fromarray((mask[y0:y1,x0:x1]*255).astype(np.uint8)).save(folder/(name+'.png'))
        row={'raw':raw,'actual_group_source':pin(path),'expected_formal_native':pin(expected_path),'camera_cache':pin(cache_path),'source_map':pin(map_path),
          'body_component_alpha_for_protection_only':pin(component_path),'protection_masks':pin(masks_path),
          'additive_RGB_max_abs_error':rgb_error,'additive_alphaT_max_abs_error':alpha_error,'full_native_RGBA_byteexact':True,
          'group_names':z['group_names'].tolist(),'ROI_xyxy':[x0,y0,x1,y1],'original_ROI_denominator':int(roi.sum()),
          'body_protection_definition':'isolated canonical alpha>.05 OR strict real V, dilate3; used only to exclude body from ground candidates, never to synthesize contributions',
          'diagnostic_colour_thresholds':{'red':'R>.2 and R>max(G,B)+.12','blue_cyan':'B>R+.03 and B>G+.015 and maxRGB>.08','dark':'maxRGB<.22 and other_native_context alphaT>.25'},
          'colour_masks_are_posthoc_diagnostics_not_truth_or_acceptance_denominators':True,
          'regions':summaries,'small_float_packet':pin(folder/'ROI_attribution.npz')}
        write(folder/'attribution.json',row);rows.append(row)
    result={'schema':'R9.actual_joint_RGB_underbody_group_attribution_CPU.v1','complete':True,'source_receipt':pin(source/'receipt.json'),'rows':rows,
      'causal_readout':'Saved RGB channels are additive premultiplied contributions from the SAME original depth-sorted raster. No layer subtraction, exposure renormalization or actor-only RGB difference.',
      'native_owner21_is_not_A9_identity_claim':True,'all_frozen_native_context_should_be_deleted':False,
      'minimal_replacement_scope':'Only ground-side residual pixels in the lower-edge ROIs, excluding canonical body/strictV/K; identify the actually contributing other-native rows before any local retirement. Preserve useful native body/context contributions.',
      'new_background_initialization':'Use only geometry-supported legal road XYZ/material subsets listed; missing XYZ is not assigned a plane or zero depth by this audit.',
      'Gaussian_rows_for_retirement_identified_by_this_group_readout':False,
      'CPU_only':True,'new_GPU_forward':0,'new_GPU_backward':0,'diffusion_calls':0,'updates':0}
    write(a.output/'attribution.json',result)
    print({'complete':True,'output':pin(a.output/'attribution.json'),'regions':[{ 'raw':x['raw'],'red':x['regions']['red_diagnostic'],'blue':x['regions']['blue_cyan_diagnostic'],'dark':x['regions']['dark_context_diagnostic']} for x in rows]})


if __name__=='__main__':main()
