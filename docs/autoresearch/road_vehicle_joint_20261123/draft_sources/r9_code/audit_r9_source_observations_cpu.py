"""Independent source RGB, semantic and geometric qualification replay, CPU only."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from audit_r9_shared_field_cpu import load,read,write,pin,gather,all4

WORK=Path(__file__).resolve().parents[2];R9=WORK/'old_raw_teacher_joint_20261123_r9';R7=WORK/'road_shared_appearance_20261122_r7'
OUT=R9/'cpu_review/joint_shared_v1';OUT.mkdir(parents=True,exist_ok=True)
sources_path=R9/'audit/raw_teacher_sources.json';sources=read(sources_path);receipt=read(R9/'world_observations_v2/receipt.json')
semantic_receipt=read(R9/'raw_road_semantic_v2/receipt.json');g=load(R7/'shared_grid_geometry.npz');seed=load(R7/'quilt_seed/atlas_seed.npz')
xyz=g['world_xyz'].astype(np.float64);domain=seed['write_domain']&g['geometry_valid'];gains={int(x['raw']):float(x['gain']) for x in read(R7/'exposure_gain.json')['records']}
rows=[]
for source in sources['sources']:
    raw=source['raw'];r=next(x for x in receipt['observations'] if x['raw']==raw);semrec=next(x for x in semantic_receipt['rows'] if x['raw']==raw)
    source_path=Path(source['raw_model']['local_path']);image=np.asarray(Image.open(source_path).convert('RGB'),np.float32)/255.
    # Independent EOTF on source samples before any interpolation.
    linear=np.where(image<=.04045,image/12.92,((image+.055)/1.055)**2.4).astype(np.float32)
    cp=R7/f'projection_cache_v1/{raw:03d}/view_cache.npz';c=load(cp);E=c['world_to_camera'];K=c['camera_K']
    camera=xyz@E[:3,:3].T+E[:3,3];z=camera[...,2];den=np.where(z>1e-6,z,1.)
    xy=camera[...,:2]/den[...,None];p=camera@K.T;p=p[...,:2]/den[...,None]-.5
    fx,fy,cx,cy,k1,k2,p1,p2,k3=c['Brown_intrinsics'];x=xy[...,0];y=xy[...,1];r2=x*x+y*y
    rad=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    xd=x*rad+2*p1*x*y+p2*(r2+2*x*x);yd=y*rad+p1*(r2+2*y*y)+2*p2*x*y
    b=np.stack([fx*xd+cx,fy*yd+cy],-1)*.6-.5
    sampled,bounds,_=gather(linear,b);sampled=sampled.astype(np.float32)
    masks_path=WORK/f'road_vehicle_joint_dggt_20261123_r8/audit/mask_coordinate_audit/rectified_masks/{raw:03d}/rectified_masks.npz';m=load(masks_path)
    surface_path=R9/f'audit/surface_visibility/{raw:03d}/surface.npz';surface=load(surface_path)
    cache_uv,_,index=gather(np.where(c['projection_valid'][...,None],c['atlas_uv_map'],0),p);cache_uv=cache_uv.astype(np.float32)
    uv_error=np.linalg.norm(cache_uv-g['uv'],axis=-1)
    geometry=all4(c['projection_valid']&c['geometry_valid']&surface['triangle_all_corners_geometry_valid'],index)
    protected=m['protected']|m['actor_visible_proxy']|m['unknown_potential']|~m['RGB_valid']
    allowed=all4(~protected,index)
    depth,_,_=gather(surface['camera_depth_m'],p);depth=depth.astype(np.float32);depth_error=abs(depth-z)
    pin_bounds=(p[...,0]>=0)&(p[...,1]>=0)&(p[...,0]<=1151)&(p[...,1]<=767)
    geometric=domain&bounds&pin_bounds&(z>1e-6)&geometry&allowed&(uv_error<=1)&(depth_error<=.03+.002*z)
    label_path=R9/f'raw_road_semantic_v2/{raw:03d}/labels_model_Brown.png';road_path=R9/f'raw_road_semantic_v2/{raw:03d}/road_eroded_model_Brown.png'
    labels=np.asarray(Image.open(label_path));road=np.asarray(Image.open(road_path))>0
    derived=binary_erosion(labels==0,iterations=2);_,_,brown_index=gather(road.astype(np.float32),b)
    semantic_good=all4(road,brown_index);expected_valid=geometric&semantic_good
    observation_path=R9/f'world_observations_v2/{raw:03d}/world_observation.npz';o=load(observation_path)
    common=(sampled*gains[raw]).astype(np.float32);common[~expected_valid]=0
    camera_center=-E[:3,:3].T@E[:3,3];ray=camera_center-xyz;ray/=np.maximum(np.linalg.norm(ray,axis=-1,keepdims=True),1e-12)
    incidence=abs((ray*g['normal_world']).sum(-1));du=np.gradient(p,axis=1);dv=np.gradient(p,axis=0)
    area=abs(du[...,0]*dv[...,1]-du[...,1]*dv[...,0]);a=np.clip(np.sqrt(area),.1,3.)
    expected_weight=np.where(expected_valid,a/(1+a)*np.clip(incidence,.05,1),0).astype(np.float32)
    colour_error=float(abs(common-o['canonical_linear_RGB']).max());weight_error=float(abs(expected_weight-o['weight']).max())
    v1path=R9/f'world_observations_v1/{raw:03d}/world_observation.npz';v1=load(v1path)
    checks={
      'raw_model_SHA_exact':pin(source_path)['sha256']==source['raw_model']['sha256']==r['raw_model_sha256']==semrec['source_sha256'],
      'view_camera_SHA_exact':pin(cp)['sha256']==r['view_cache_sha256'],
      'R8_mask_SHA_exact':pin(masks_path)['sha256']==r['mask_sha256'],
      'surface_visibility_SHA_exact':pin(surface_path)['sha256']==r['surface_visibility_sha256'],
      'semantic_labels_SHA_exact':pin(label_path)['sha256']==semrec['labels_sha256'],
      'semantic_road_SHA_exact':pin(road_path)['sha256']==semrec['road_mask_sha256']==r['semantic_road_sha256'],
      'observations_SHA_exact':pin(observation_path)['sha256']==r['observation_sha256'],
      'road0_erode2_exact':bool(np.array_equal(road,derived)),
      'pre_semantic_geometry_domain_exact':bool(np.array_equal(geometric,o['pre_semantic_geometric_candidate'])),
      'all_original_qualified_valid_texels_exact':bool(np.array_equal(expected_valid,o['valid_mask'])),
      'source_RGB_replay_within_1e_7':colour_error<=1e-7,'source_weight_replay_within_1e_7':weight_error<=1e-7,
      'all_accepted_semantic_all4_road':bool(semantic_good[expected_valid].all()),
      'all_accepted_surface_all4_valid':bool(geometry[expected_valid].all()),
      'all_accepted_new_protection_all4_free':bool(allowed[expected_valid].all()),
      'all_accepted_UV_le_1':bool((uv_error[expected_valid]<=1).all()),
      'all_accepted_depth_in_contract':bool((depth_error[expected_valid]<=.03+.002*z[expected_valid]).all()),
      'real_anchor_count_exact':int((expected_valid&seed['same_world_lock']).sum())==r['candidate_real_anchor_texels'],
      'zero_sentinel_outside_valid':not bool(o['canonical_linear_RGB'][~o['valid_mask']].any())}
    rows.append({'raw':raw,'checks':checks,'source_raw_model':pin(source_path),'qualified_observations':pin(observation_path),
      'old_v1_observation_retained':pin(v1path),'old_v1_candidate_texels':int(v1['valid_mask'].sum()),
      'original_world_scope_texels':int(domain.sum()),'pre_semantic_geometric_candidate':int(geometric.sum()),
      'semantic_nonroad_removed':int((geometric&~semantic_good).sum()),'qualified_texels':int(expected_valid.sum()),
      'real_anchor_texels':int((expected_valid&seed['same_world_lock']).sum()),
      'RGB_replay_max_abs_error':colour_error,'weight_replay_max_abs_error':weight_error,
      'model_Brown_coordinate_max_abs_export_error_px':float(abs(b.astype(np.float32)-o['model_Brown_xy']).max()),
      'pinhole_coordinate_max_abs_export_error_px':float(abs(p.astype(np.float32)-o['native_pinhole_xy']).max()),
      'semantic_is_frozen_model_proxy_not_GT':True,'old_W_CS_used':False,'hard_final_or_raw_native_RGB_used':False})
result={'schema':'R9.independent_source_sampling_qualification_audit.v1','complete':True,
  'decision':'PASS' if all(all(x['checks'].values()) for x in rows) else 'FAIL',
  'source_manifest':pin(sources_path),'world_observation_receipt':pin(R9/'world_observations_v2/receipt.json'),
  'semantic_receipt':pin(R9/'raw_road_semantic_v2/receipt.json'),'rows':rows,
  'CPU_only':True,'new_GPU_forward':0,'new_diffusion_calls':0,'native_F':0,'native_B':0,'Gaussian_updates':0}
write(OUT/'source_sampling_qualification_audit.json',result)
print(json.dumps(result))
