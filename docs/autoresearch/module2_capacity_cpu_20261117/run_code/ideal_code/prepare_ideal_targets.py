"""仅CPU准备理想教师；复用独立mesh ray maps，不调用GS渲染或训练。"""
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from ideal_world_teacher import evaluate_world, srgb_to_linear, sampling_qualification, plane_ray_jacobian

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAMPAIGN/'road_asset_20261116_r1'
ROOT=CAMPAIGN/'road_capacity_cpu_20261117_r1'
FRAMES=(86,110,134,179,198)


def load(path, keys):
    with np.load(path,allow_pickle=False) as source:
        return {key:source[key] for key in keys}


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path=Path(path)
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def quantiles(value):
    value=np.asarray(value);value=value[np.isfinite(value)]
    return None if not len(value) else np.quantile(value,[0,.25,.5,.75,1]).tolist()


def camera(bank,frame_id,observation):
    rows=(bank/'calib'/f'{frame_id}.txt').read_text().splitlines()
    calib={row.split(':')[0]:np.fromstring(row.split(':')[1],sep=' ') for row in rows}
    K=calib['P0'].reshape(3,4)[:,:3].copy();K[:2]*=480/observation['camera_size'][0]
    matrix=np.eye(4);matrix[:3]=calib['Tr_velo_to_cam_0'].reshape(3,4)
    matrix=matrix@np.linalg.inv(np.asarray(observation['ego_world']).reshape(4,4))
    inverse=np.linalg.inv(matrix);yy,xx=np.mgrid[:320,:480]
    transform=np.linalg.inv(K).T@inverse[:3,:3].T
    rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],axis=-1)@transform
    return K,matrix,inverse[:3,3],rays,transform[0],transform[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'ideal_targets_r2')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    asset=OLD/'phase2/formal_r1/asset';meshdir=OLD/'phase0/supervision_triangle_r8'
    fixed=CAMPAIGN/'scene065_closedloop_20261116_r1/handoff_coverage_r2'
    config=read(CAMPAIGN/'scene065_material_loop_20261114_r1/config.json');bank=Path(config['bank'])
    ids={row['raw']:row['id'] for row in read(bank/'receipt.json')['frames']}
    road=load(asset/'road.npz',('active','minimum_model_depth','meter_to_model','road_opacity','gaussian_sigma_m','reference_srgb','model_xyz','model_from_world'))
    prior=load(asset/'provenance.npz',('guarded_world_rows','surface_supported','legal_write'))
    atlas=load(asset/'atlas.npz',('surface_world_xyz','grid_world_row'))
    grid=atlas['grid_world_row'];surface=np.empty((len(road['active']),3));surface[grid.ravel()]=atlas['surface_world_xyz'].reshape(-1,3)
    all_xyz=[]
    for raw in FRAMES:
        m=load(meshdir/f'{raw:03d}.npz',('ground_world_xyz','frozen_evaluation_domain'))
        all_xyz.append(m['ground_world_xyz'][m['frozen_evaluation_domain']&np.isfinite(m['ground_world_xyz']).all(2)])
    origin=np.median(np.concatenate(all_xyz),axis=0)[:2]
    es=np.mean(np.diff(atlas['surface_world_xyz'][...,:2],axis=1),axis=(0,1));es/=np.linalg.norm(es)
    el=np.array([-es[1],es[0]])
    field=dict(schema='ideal_world_road_teacher.v1',world_frame=dict(origin_xy=origin.tolist(),longitudinal_xy=es.tolist(),lateral_xy=el.tolist(),
        origin_rule='componentwise median canonical mesh hits in five historical fixed domains; no RGB used'),
        curve=dict(kind='circle arc',radius_m=120.,coordinates='u=R atan2(s,R-l), v=R-hypot(s,R-l); width is world-XY outer paint width'),
        asphalt=dict(constant_srgb=[.32,.33,.35],lowfreq=dict(amplitude_srgb=[.015,.010,.008],longitudinal_period_m=8.,lateral_period_m=2.)),
        paint_srgb=[.88,.88,.84],sampling=dict(main='pixel-center canonical G_s mesh hit; plane fallback separately tagged',
        detail_minimum_pixels=3.,nyquist_cycles_per_axis_maximum=.5,main_domain_never_shrunk=True,
        pixel_footprint_supersampling='4x4 spot-check plan only; no full AA target is claimed',
        marking_failure_requires_sampling_convention_review=True),frames=list(FRAMES),render_wh=[480,320],
        role='all-five-view ideal capacity oracle, not generalization; actual-data experiment remains two train and three diagnose',
        color='native sRGB appearance; fixed-alpha premult RGB target; no intrinsic-radiance claim')
    cases=[dict(name='constant',asphalt='constant',gate_role='primary',marking=None),
        dict(name='lowfreq',asphalt='lowfreq',gate_role='primary',marking=None)]
    for width in (.12,.24):
        cases.append(dict(name='marking_w'+str(int(width*1000)).zfill(3),asphalt='lowfreq',gate_role='resolved_detail_only',
            marking=dict(width_m=width,cross_edge_m=.02,period_m=3.,on_m=1.5,phase_m=.45,longitudinal_edge_m=.10,profile='flat top with cubic smoothstep edges')))
    cases.append(dict(name='stress_highfreq',asphalt='stress',gate_role='diagnostic_only_never_capability_failure',
        marking=dict(width_m=.025,cross_edge_m=.004,period_m=.10,on_m=.05,phase_m=.012,longitudinal_edge_m=.01,profile='flat top with cubic smoothstep edges')))
    field['cases']=cases
    (out/'field_config.json').write_text(json.dumps(field,indent=2)+'\n')
    for case in cases:(out/case['name']).mkdir()
    (out/'geometry').mkdir()
    active=np.flatnonzero(road['active']);guardpath=CAMPAIGN/'road_direct_teacher_20261116_r1/phase0/lifecycle_r4/save_color/row_scopes.npz'
    guard=load(guardpath,('active_world_rows','conservative_K_guard'))
    if not np.array_equal(active,guard['active_world_rows']):raise ValueError('archived guard active-row mapping differs')
    fullguard=prior['guarded_world_rows'].copy();fullguard[active]|=guard['conservative_K_guard']
    primary=road['active']&prior['surface_supported']&prior['legal_write']&~fullguard&(road['minimum_model_depth']<=0)
    np.savez_compressed(out/'color_scopes.npz',active_world_rows=active,primary_world=primary,primary_active=primary[active],
        conditional_relaxed_world=road['active'],conditional_relaxed_active=np.ones(len(active),bool),K_guard_world=fullguard,
        original_K_guard_world=prior['guarded_world_rows'],c0_srgb=road['reference_srgb'][active],
        fixed_road_opacity=road['road_opacity'][active],fixed_sigma_m=road['gaussian_sigma_m'][active])
    plane_path=CAMPAIGN/'waymo_val065_road_completion_20261111_r1/geometry_no_colour/geometry.npz'
    plane=load(plane_path,('plane','coordinate_anchor'));a,b,c=plane['plane']
    normal4=np.linalg.inv(plane['coordinate_anchor']).T@np.array([-a,-b,1.,-c]);normal4/=np.linalg.norm(normal4[:3])
    rows=[];pins=[pin(asset/'road.npz'),pin(asset/'provenance.npz'),pin(asset/'atlas.npz'),pin(guardpath),pin(plane_path),
                  pin(Path(__file__).resolve()),pin(Path(__file__).with_name('ideal_world_teacher.py').resolve())]
    for raw in FRAMES:
        keys=('ground_world_xyz','triangle_world_rows','triangle_weights','surface_intersection_valid','geometry_support','K','atlas_uv','frozen_evaluation_domain')
        m=load(meshdir/f'{raw:03d}.npz',keys);D=m['frozen_evaluation_domain'].copy()
        historical=load(fixed/f'contribution_diagnostics_20261116/float/{raw:03d}.npz',('domain',))['domain']
        if not np.array_equal(D,historical):raise ValueError('historical main evaluation changed')
        baseline_path=asset/f'../rendered/{raw:03d}/arrays.npz'
        baseline=load(baseline_path,('road_only_alpha','road_only_rgb'))
        alpha=baseline['road_only_alpha'].squeeze().astype(np.float32)
        obs_path=bank/'observations'/f'{raw:03d}.json';obs=read(obs_path)
        K,camera_matrix,camera_origin,rays,du,dv=camera(bank,ids[raw],obs)
        xyz=m['ground_world_xyz'].copy();mesh_valid=np.isfinite(xyz).all(2)&m['surface_intersection_valid']
        denominator=rays@normal4[:3]
        depth_plane=-(camera_origin@normal4[:3]+normal4[3])/denominator
        fallback=D&~mesh_valid&np.isfinite(depth_plane)&(depth_plane>.2)
        xyz[fallback]=camera_origin+depth_plane[fallback,None]*rays[fallback]
        defined=mesh_valid|fallback
        if not defined[D].all():raise ValueError('main domain still has undefined ideal target')
        kind=np.zeros(D.shape,np.uint8);kind[mesh_valid]=1;kind[fallback]=2
        normals=np.zeros_like(xyz);valid_tri=mesh_valid&(m['triangle_world_rows']>=0).all(2)
        verts=surface[m['triangle_world_rows'][valid_tri]]
        face_normal=np.cross(verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]);face_normal/=np.linalg.norm(face_normal,axis=1)[:,None]
        normals[valid_tri]=face_normal;normals[fallback]=normal4[:3]
        jac=np.full((*D.shape,2,2),np.nan)
        jac[defined]=plane_ray_jacobian(camera_origin,rays[defined],du,dv,normals[defined],xyz[defined])
        near=np.zeros(D.shape,bool)
        zverts=verts@camera_matrix[:3,:3].T+camera_matrix[:3,3]
        mins=road['minimum_model_depth'][m['triangle_world_rows'][valid_tri]]/float(road['meter_to_model'])
        near[valid_tri]=np.any(zverts[:,:,2]<=np.maximum(.2,mins),axis=1)
        plane_error=np.full(D.shape,np.nan);plane_error[mesh_valid]=xyz[mesh_valid]@normal4[:3]+normal4[3]
        uvh=xyz[defined]@camera_matrix[:3,:3].T+camera_matrix[:3,3];uvh=uvh@K.T
        uv=uvh[:,:2]/uvh[:,2,None];yy,xx=np.nonzero(defined)
        projection_error=float(np.abs(uv-np.c_[xx+.5,yy+.5]).max())
        boundary=D&~binary_erosion(D,iterations=2)
        np.savez_compressed(out/'geometry'/f'{raw:03d}.npz',surface_world_xyz=xyz,surface_reference_kind=kind,
            teacher_defined=defined,canonical_mesh_hit_valid=mesh_valid,plane_fallback=fallback,
            triangle_world_rows=m['triangle_world_rows'],triangle_weights=m['triangle_weights'],pixel_world_xy_jacobian=jac,
            camera_K=K,world_to_camera=camera_matrix,fixed_evaluation_domain=D,K=m['K'],fixed_alpha=alpha,
            baseline_forward_rgb_srgb__not_teacher=baseline['road_only_rgb'],near_policy_excluded=near,
            fixed_domain_boundary=boundary,geometry_support=m['geometry_support'],plane_signed_residual_m=plane_error)
        safe=np.where(defined[...,None],xyz,np.r_[origin,0.])
        cases_summary=[]
        for case in cases:
            rgb,paint=evaluate_world(safe,case,field);rgb[~defined]=0;paint[~defined]=0
            sampling=sampling_qualification(safe,jac,case,field)
            detail=D&defined&sampling['line_corridor']&sampling['resolvable']
            np.savez_compressed(out/case['name']/f'{raw:03d}.npz',target_rgb_srgb=rgb.transpose(2,0,1).astype(np.float32),
                target_rgb_linear=srgb_to_linear(rgb).transpose(2,0,1).astype(np.float32),
                target_premult_srgb=(rgb*alpha[...,None]).transpose(2,0,1).astype(np.float32),
                fixed_evaluation_domain=D,mask=D,q=D.astype(np.float32),fixed_alpha=alpha,teacher_defined=defined,
                surface_reference_kind=kind,plane_fallback=fallback,paint_fraction=paint.astype(np.float32),K=m['K'],
                detail_sampling_qualified=sampling['resolvable'],detail_evaluation_mask=detail,
                width_px=sampling['width_px'].astype(np.float32),period_px=sampling['period_px'].astype(np.float32),
                on_px=sampling['on_px'].astype(np.float32),off_px=sampling['off_px'].astype(np.float32),nyquist_ok=sampling['nyquist_ok'],
                asphalt_max_cycles_per_pixel_axis=sampling['asphalt_max_cycles_per_pixel_axis'].astype(np.float32),
                stress_only=np.asarray(case['gate_role'].startswith('diagnostic')))
            Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(out/case['name']/f'{raw:03d}_ideal.png')
            corridor=D&sampling['line_corridor']
            cases_summary.append(dict(case=case['name'],main_pixels=int(D.sum()),paint_pixels=int((D&(paint>.01)).sum()),
                resolved_detail_pixels=int(detail.sum()),undersampled_line_pixels=int((corridor&~sampling['resolvable']).sum()),
                asphalt_nyquist_exceeded_pixels=int((D&(sampling['asphalt_max_cycles_per_pixel_axis']>.5)).sum()),
                width_px_quantiles=quantiles(sampling['width_px'][corridor]),period_px_quantiles=quantiles(sampling['period_px'][corridor]),gate_role=case['gate_role']))
        # Center-triangle qualification is diagnostic, not exact Gaussian alphaT contribution.
        center_guard=np.zeros(D.shape,bool);center_guard[valid_tri]=np.any(fullguard[m['triangle_world_rows'][valid_tri]],axis=1)
        rows.append(dict(raw=raw,main_pixels=int(D.sum()),K_pixels=int(m['K'].sum()),mesh_pixels=int((D&mesh_valid).sum()),
            fallback_pixels=int(fallback.sum()),source_coverage_gap=int((D&(alpha<.05)).sum()),source_thin=int((D&(alpha>=.05)&(alpha<=.9)).sum()),
            source_supported=int((D&(alpha>.9)).sum()),near_policy_center_triangles=int((D&near).sum()),
            fixed_center_triangle_guard_pixels=int((D&center_guard).sum()),center_guard_is_not_actual_contribution=True,
            reprojection_max_px=projection_error,plane_residual_mesh_quantiles_m=quantiles(plane_error[D&mesh_valid]),cases=cases_summary))
        pins.extend([pin(meshdir/f'{raw:03d}.npz'),pin(baseline_path),pin(obs_path),pin(bank/'calib'/f'{ids[raw]}.txt')])
        print(json.dumps(rows[-1]),flush=True)
    receipt=dict(schema='cpu_ideal_capacity_preparation.v1',status='prepared_cpu_only',gpu_executed=False,cuda_imported=False,
        actual_GS_render_executed=False,training_updates=0,gpu_auto_start=False,threads=1,
        fixed_main_pixel_frames=sum(row['main_pixels'] for row in rows),fixed_K_pixel_frames=sum(row['K_pixels'] for row in rows),
        plane_fallback_pixels=sum(row['fallback_pixels'] for row in rows),rows=rows,source_pins=pins,
        source_asset=str(asset),world_field_config=str(out/'field_config.json'),main_targets_from_GS_RGB=False,
        scopes=dict(primary_rows=int(primary.sum()),conditional_relaxed_rows=len(active),original_guard_world_rows=int(prior['guarded_world_rows'].sum()),
                    archived_conservative_active_guard=int(guard['conservative_K_guard'].sum()),union_guard_world_rows=int(fullguard.sum()),
                    conditional_relaxed_role='road-only capacity diagnosis; never production or keep preservation'),
        fixed_contribution=dict(baseline_forward_saved=True,exact_frozen_b_not_evaluated=True,
            next_native_operator='P(c)=P0+A_E(c-c0_E); b=P0-A_E*c0_E. Baseline RGB is offset only, never target.',
            actual_native_forward_adjoint_linearity_and_frozen_b='pending explicitly authorized GPU; CPU eligibility is not alphaT proof'),
        failure_ledger_delta='none; preparation only',human_verdict=None)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    (out/'freeze.json').write_text(json.dumps(dict(schema='immutable_ideal_capacity_targets.v1',source_files=pins,
        field=pin(out/'field_config.json'),targets=[pin(out/case['name']/f'{raw:03d}.npz') for case in cases for raw in FRAMES],
        geometry=[pin(out/'geometry'/f'{raw:03d}.npz') for raw in FRAMES],scopes=pin(out/'color_scopes.npz'),gpu_executed=False),indent=2)+'\n')
    print(json.dumps(dict(completed=True,main=receipt['fixed_main_pixel_frames'],fallback=receipt['plane_fallback_pixels'],gpu_executed=False)))


if __name__=='__main__':main()
