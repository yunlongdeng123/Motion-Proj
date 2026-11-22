"""Disentangle cached geometry, UV support and appearance permission; read-only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def pin(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();C=a.campaign_root
    old=C/'road_vehicle_joint_dggt_20261123_r8/residual_rgb_groups_v1/CPU_attribution_v2_focus'
    field_path=C/'old_raw_teacher_joint_20261123_r9/joint_shared_continuous_v2/final_atlas.npz';field=load(field_path)
    grid_path=C/'road_shared_appearance_20261122_r7/shared_grid_geometry.npz';g=load(grid_path);H,W=field['source_kind'].shape;rows=[]
    a.output.mkdir(parents=True,exist_ok=True)
    for raw in [86,110]:
        original=read(old/f'{raw:03d}/attribution.json');small=load(old/f'{raw:03d}/ROI_attribution.npz');x0,y0,x1,y1=original['ROI_xyxy']
        cache_path=C/f'road_shared_appearance_20261122_r7/projection_cache_v1/{raw:03d}/view_cache.npz';v=load(cache_path)
        source_path=Path(str(v['source_map_path']));source=load(source_path)
        tri_path=C/f'old_raw_teacher_joint_20261123_r9/audit/surface_visibility/{raw:03d}/surface.npz';t=load(tri_path)
        uv=v['atlas_uv_map'];finite=np.isfinite(uv).all(-1);safe=np.nan_to_num(uv)
        u=np.floor(safe[...,0]).astype(int);y=np.floor(safe[...,1]).astype(int)
        bounds=finite&(u>=0)&(y>=0)&(u+1<W)&(y+1<H);u=np.clip(u,0,W-2);y=np.clip(y,0,H-2)
        def corners(m):return [m[y,u],m[y,u+1],m[y+1,u],m[y+1,u+1]]
        def all4(m):return bounds&np.logical_and.reduce(corners(m))
        def any4(m):return bounds&np.logical_or.reduce(corners(m))
        flags={'view_geometry_valid':v['geometry_valid'],'view_projection_valid':v['projection_valid'],
          'source_triangle_all3_geometry_valid':t['triangle_all_corners_geometry_valid'],
          'atlas_UV_finite':finite,'atlas_UV_all4_in_bounds':bounds,
          'atlas_geometry_all4_valid':all4(g['geometry_valid']),
          'field_permission_all4':all4(field['permission_mask']),'field_permission_any4':any4(field['permission_mask']),
          'field_sourcekind_all4_positive':all4(field['source_kind']>0),
          'field_RGB_all4_finite':all4(np.isfinite(field['atlas_linear']).all(-1))}
        partition={};remaining=np.ones(finite.shape,bool)
        failures=[('no_cached_view_geometry',~flags['view_geometry_valid']),
          ('view_geometry_projection_invalid',~flags['view_projection_valid']),
          ('atlas_query_not_finite',~finite),('atlas_query_bilinear_outside',~bounds),
          ('source_triangle_invalid_corner',~flags['source_triangle_all3_geometry_valid']),
          ('atlas_quad_geometry_incomplete',~flags['atlas_geometry_all4_valid']),
          ('entirely_outside_field_permission',~flags['field_permission_any4']),
          ('field_permission_bilinear_boundary',~flags['field_permission_all4']),
          ('field_sourcekind_unknown',~flags['field_sourcekind_all4_positive']),
          ('field_RGB_nonfinite',~flags['field_RGB_all4_finite'])]
        for name,condition in failures:partition[name]=remaining&condition;remaining&=~condition
        partition['complete_geometry_and_material_support']=remaining
        union=np.zeros_like(remaining)
        for value in partition.values():assert not (union&value).any();union|=value
        assert union.all()
        regions={}
        for name in ['original_proxy_lower_edge_ROI','red_diagnostic','blue_cyan_diagnostic','dark_context_diagnostic','focused_ground_envelope','focused_residual_candidate']:
            mask=np.zeros_like(remaining);mask[y0:y1,x0:x1]=small[name]
            independent={k:int((mask&v).sum()) for k,v in flags.items()}
            counts={k:int((mask&v).sum()) for k,v in partition.items()};assert sum(counts.values())==int(mask.sum())
            regions[name]={'fixed_pixels':int(mask.sum()),'independent_true_counts':independent,
              'query_invalid_appearance_checks_not_applicable':int((mask&~bounds).sum()),
              'appearance_conditional_on_valid_query':{k:{'query_valid_pixels':int((mask&bounds).sum()),'passed':int((mask&v).sum()),'failed_with_query_valid':int((mask&bounds&~v).sum())} for k,v in flags.items() if k.startswith('field_')},
              'disjoint_first_failure_partition':counts,'partition_complete':True,
              'old_source_map_query_mask_count_diagnostic':int((mask&source['query_mask']).sum()) if 'query_mask' in source else None,
              'old_source_requested_count_diagnostic':int((mask&source['requested']).sum()) if 'requested' in source else None}
        np.savez_compressed(a.output/f'{raw:03d}_full_support_flags.npz',**flags,**{'partition_'+k:v for k,v in partition.items()})
        row={'raw':raw,'original_attribution':pin(old/f'{raw:03d}/attribution.json'),'camera_cache':pin(cache_path),'source_map':pin(source_path),
          'surface_triangle_certificate':pin(tri_path),'regions':regions,
          'source_and_view_geometry_masks_exact':bool(np.array_equal(source['geometry_valid'],v['geometry_valid']))}
        write(a.output/f'{raw:03d}_support_partition.json',row);rows.append(row)
    result={'schema':'R10.cached_surface_geometry_vs_appearance_support_partition.v1','complete':True,
      'grid':pin(grid_path),'field':pin(field_path),'rows':rows,'all_original_denominators_retained':True,
      'cached_view_geometry_missing_is_not_physical_ground_absence':True,'appearance_permission_failure_is_not_called_geometry_failure':True,
      'original_g_or_field_modified':False,'CPU_only':True,'GPU_forward':0,'updates':0}
    write(a.output/'support_partition.json',result);print({'receipt':pin(a.output/'support_partition.json'),'red':[{'raw':r['raw'],**r['regions']['red_diagnostic']} for r in rows]})

if __name__=='__main__':main()
