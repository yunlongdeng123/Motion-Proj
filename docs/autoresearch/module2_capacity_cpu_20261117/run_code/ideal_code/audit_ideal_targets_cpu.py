"""独立CPU检查理想教师世界位置、冻结主域、相机导数、标线定义和来源。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from ideal_world_teacher import evaluate_world, curve_coordinates

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--targets',type=Path,default=ROOT/'ideal_targets_r2')
    p.add_argument('--output',type=Path,default=ROOT/'qa/ideal_targets_cpu_acceptance.json');args=p.parse_args()
    field=json.loads((args.targets/'field_config.json').read_text());frozen=json.loads((args.targets/'freeze.json').read_text())
    receipt=json.loads((args.targets/'receipt.json').read_text());checks={};rows=[]
    def check(name,value):checks[name]=bool(value)
    def load(path):
        with np.load(path,allow_pickle=False) as saved:return {key:saved[key] for key in saved.files}
    with np.load(Path(receipt['source_asset'])/'atlas.npz') as source:
        grid=source['grid_world_row'];surface=np.empty((int(grid.max())+1,3))
        surface[grid.ravel()]=source['surface_world_xyz'].reshape(-1,3)
    allpins=frozen['source_files']+[frozen['field'],frozen['scopes']]+frozen['targets']+frozen['geometry']
    check('all_actual_sources_and_targets_frozen',all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in allpins))
    frame=field['world_frame'];es=np.asarray(frame['longitudinal_xy']);el=np.asarray(frame['lateral_xy']);origin=np.asarray(frame['origin_xy'])
    check('world_frame_metric_orthonormal',np.max(np.abs(np.array([es,el])@np.array([es,el]).T-np.eye(2)))<1e-12 and np.linalg.det(np.array([es,el]))>0)
    radius=field['curve']['radius_m']
    for case in field['cases']:
        if not case['marking']:continue
        m=case['marking'];u=m['phase_m']+np.arange(-3,4)*m['period_m'];phi=u/radius
        def points(v):
            r=radius-v;s=r*np.sin(phi);l=radius-r*np.cos(phi)
            return np.c_[origin+s[:,None]*es+l[:,None]*el,np.zeros(len(u))]
        _,paint=evaluate_world(points(0),case,field)
        uu,vv,_,_=curve_coordinates(points(0),field)
        check(case['name']+'_known_arc_length_phase',np.max(np.abs(uu-u))<1e-9 and np.max(np.abs(vv))<1e-9 and np.min(paint)>1-1e-9)
        _,edge=evaluate_world(points(m['width_m']/2),case,field)
        _,half=evaluate_world(points(m['width_m']/2-m['cross_edge_m']/2),case,field)
        check(case['name']+'_known_width_and_softedge',np.max(edge)<1e-8 and np.max(np.abs(half-.5))<1e-8)
    main=keep=fallback_total=0
    for raw in field['frames']:
        geometry=load(args.targets/'geometry'/f'{raw:03d}.npz');D=geometry['fixed_evaluation_domain'];kind=geometry['surface_reference_kind']
        alpha=geometry['fixed_alpha'];xyz=geometry['surface_world_xyz'];main+=int(D.sum());keep+=int(geometry['K'].sum());fallback_total+=int((D&(kind==2)).sum())
        check(str(raw)+'_complete_main_domain_target',geometry['teacher_defined'][D].all() and np.isfinite(xyz[D]).all())
        check(str(raw)+'_fallback_flags_explicit',np.array_equal(geometry['plane_fallback'],kind==2))
        K=geometry['camera_K'];C=geometry['world_to_camera'];inv=np.linalg.inv(C);transform=np.linalg.inv(K).T@inv[:3,:3].T
        yy,xx=np.nonzero(D&(kind==1));take=np.linspace(0,len(xx)-1,min(64,len(xx))).astype(int);x=xx[take];y=yy[take]
        pts=xyz[y,x];ray=np.c_[x+.5,y+.5,np.ones(len(x))]@transform
        # 独立用原面片顶点和偏移光线做有限差分，不调用teacher Jacobian函数。
        Jxy=geometry['pixel_world_xy_jacobian'][y,x]
        origin_cam=inv[:3,3]
        vertices=surface[geometry['triangle_world_rows'][y,x]]
        normals=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
        plane_d=-np.einsum('ij,ij->i',normals,vertices[:,0])
        bary=geometry['triangle_weights'][y,x]
        check(str(raw)+'_mesh_barycentric_reconstructs',np.max(np.abs((vertices*bary[...,None]).sum(1)-pts))<1e-8)
        derivative_errors=[]
        eps=1e-3
        for axis in range(2):
            hits=[]
            for sign in (-1,1):
                target_uv=np.c_[x+.5,y+.5];target_uv[:,axis]+=sign*eps
                shifted_ray=np.c_[target_uv,np.ones(len(x))]@transform
                t=-(normals@origin_cam+plane_d)/np.einsum('ij,ij->i',normals,shifted_ray)
                hits.append(origin_cam+shifted_ray*t[:,None])
            finite_difference=(hits[1][:,:2]-hits[0][:,:2])/(2*eps)
            derivative_errors.append(float(np.max(np.abs(finite_difference-Jxy[:,axis]))))
        check(str(raw)+'_pixel_world_J_finite_difference',max(derivative_errors)<1e-4)
        case_rows=[]
        for case in field['cases']:
            target=load(args.targets/case['name']/f'{raw:03d}.npz');P=target['target_rgb_srgb']
            check(str(raw)+'_'+case['name']+'_fixed_D_q',np.array_equal(target['mask'],D) and np.array_equal(target['q'],D.astype(np.float32)))
            check(str(raw)+'_'+case['name']+'_valid_RGB',np.isfinite(P[:,D]).all() and np.all((P[:,D]>=0)&(P[:,D]<=1)))
            check(str(raw)+'_'+case['name']+'_fixed_alpha_target',np.array_equal(target['fixed_alpha'],alpha)
                  and np.max(np.abs(target['target_premult_srgb']-P*alpha[None]))<2*np.finfo(np.float32).eps)
            expected,paint=evaluate_world(xyz[D],case,field)
            check(str(raw)+'_'+case['name']+'_world_function_reconstructs',np.max(np.abs(expected-P[:,D].T))<np.finfo(np.float32).eps)
            if case['name']=='constant':check(str(raw)+'_constant_exact_fixture',np.max(np.abs(P[:,D].T-np.asarray(field['asphalt']['constant_srgb'])))<np.finfo(np.float32).eps)
            detail=target['detail_evaluation_mask']
            if case['marking']:
                check(str(raw)+'_'+case['name']+'_detail_3pixel_gate',np.all(target['width_px'][detail]>=3-1e-6)
                    and np.all(target['on_px'][detail]>=3-1e-6) and np.all(target['off_px'][detail]>=3-1e-6))
            if case['gate_role'].startswith('diagnostic'):check(str(raw)+'_stress_is_not_failure_gate',bool(target['stress_only']) and not detail.any())
            case_rows.append(dict(case=case['name'],detail_pixels=int(detail.sum()),paint_pixels=int((D&(target['paint_fraction']>.01)).sum())))
        rows.append(dict(raw=raw,main_pixels=int(D.sum()),fallback_pixels=int((D&(kind==2)).sum()),camera_J_consistency_error=max(derivative_errors),cases=case_rows))
    check('historical_38416_preserved',main==38416);check('keep_77266_preserved',keep==77266);check('only_218_plane_fallbacks',fallback_total==218)
    check('no_torch_native_or_cuda_import',not any(name.startswith(('torch','cupy','diff_gaussian_rasterization')) for name in sys.modules))
    check('no_render_train_GPU_in_receipt',not receipt['gpu_executed'] and not receipt['actual_GS_render_executed'] and receipt['training_updates']==0 and not receipt['gpu_auto_start'])
    report=dict(schema='independent_ideal_teacher_CPU_acceptance.v1',CPU_preparation_status='pass' if all(checks.values()) else 'failed',
        actual_capacity_status='not_evaluated_requires_user_GPU_authorization',checks=checks,frames=rows,
        GPU_executed=False,real_scene_render_calls=0,actual_model_training_updates=0,background_controller_created=False,
        process_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,threads=1,human_verdict=None,
        limitations=['pixel-center main teacher; full independent AA validation not yet performed',
                     '218 plane fallbacks remain separately tagged in the unchanged main domain',
                     'native linearity/adjoint/fixed b/box-LS residual and K remain unevaluated',
                     'CPU toy certificate is not real-road capacity evidence'])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['CPU_preparation_status'],checks=len(checks),failed=[k for k,v in checks.items() if not v],peak_MiB=report['process_peak_RSS_MiB'],GPU_executed=False)))


if __name__=='__main__':main()
