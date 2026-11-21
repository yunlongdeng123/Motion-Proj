"""Read-only 512px perspective covariance and color-neighborhood diagnostics."""
from pathlib import Path
import hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent

def load(path):return dict(np.load(path,allow_pickle=False))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stats(a,unit='px'):
    a=np.asarray(a,float)
    if not len(a):return {'count':0}
    result={'count':int(a.size),'unit':unit,'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),
            'p95':float(np.quantile(a,.95)),'mean':float(a.mean()),
            }
    if unit=='px':result.update(fraction_gt_1px=float((a>1).mean()),fraction_gt_2px=float((a>2).mean()))
    return result
def sigmas(cov):
    ev=np.linalg.eigvalsh(cov)
    return np.sqrt(np.maximum(ev[:,::-1],0))
def covariance(a):
    q=a.get('rotation',a.get('rotation_wxyz')).astype(float)
    R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    return np.einsum('nik,nk,njk->nij',R,a['scales'].astype(float)**2,R)
def projected(a,cov,spec,cache):
    W=np.asarray(spec['camera_from_actor']);pc=a['xyz']@W[:3,:3].T+W[:3,3]
    f=256*np.asarray(spec['projection_matrix'])[0,0]
    z=pc[:,2];uv=f*pc[:,:2]/z[:,None]+255.5
    J=np.zeros((len(z),2,3));J[:,0,0]=f/z;J[:,1,1]=f/z
    J[:,0,2]=-f*pc[:,0]/z**2;J[:,1,2]=-f*pc[:,1]/z**2
    A=J@W[:3,:3]
    C=A@cov@np.transpose(A,(0,2,1))
    pixel=np.rint(uv).astype(int)
    inside=(z>.1)&(pixel[:,0]>=0)&(pixel[:,0]<512)&(pixel[:,1]>=0)&(pixel[:,1]<512)
    ids=np.flatnonzero(inside);xy=pixel[ids]
    near=np.zeros(len(z),bool)
    near[ids]=(cache['alpha'][xy[:,1],xy[:,0]]>=.9)&(np.abs(z[ids]-cache['depth'][xy[:,1],xy[:,0]])<=.075)
    return uv,z,C,A,near

def main():
    started=time.time()
    paths={'teacher':ROOT/'source_teacher/gaussians_activated_registered_actor.npz',
           'student_initial':ROOT/'assets_object_clean_color_r1/student_init.npz',
           'student_geometry40':ROOT/'geometry_object_r1/actor.npz'}
    assets={k:load(p) for k,p in paths.items()}
    covs={k:covariance(a) for k,a in assets.items()}
    source=load(ROOT/'assets_object_clean_color_r1/generated_DC_source.npz')
    specs=json.loads((ROOT/'assets_object_clean_color_r1/camera_specs.json').read_text())['synthetic']
    init=assets['student_initial'];teacher=assets['teacher']
    teacher_dc=teacher['features_dc'][:,0]
    reconstructed=np.einsum('nk,nkc->nc',source['weights'],teacher_dc[source['teacher_row_ids']])
    result={'schema':'a20.same_camera_detail_footprint.v1','gpu_used':False,'assets_modified':False,
            'source_files':{k:{'path':str(p.resolve()),'sha256':sha(p)} for k,p in paths.items()},
            'method':{
              'camera':'f=512*P00/2; u,v=f*(X/Z,Y/Z)+(511/2); same cached positive-FOV synthetic cameras',
              'covariance':'world Sigma=R diag(activated scales^2) R^T; C2D=J Rcam Sigma Rcam^T J^T',
              'visible_near_surface':'mean center rounds inside512, z>.1m, teacher cached alpha>=.90 and abs(z-cached normalized native teacher depth)<=.075m',
              'node_subset':'opacity>=.10 plus above common teacher-depth near-surface criterion; unweighted per-node quantiles',
              'renderer_lowpass':'Report both C2D and C2D+0.3*I px^2; sigma is sqrt eigenvalue, not 0.3px',
              'density':'near-surface qualified center count per1000 teacher alpha>=.90 pixels; rounded center-bin multiplicity and nearest projected center distance',
              'color_neighborhood':'exact recorded positive source weights; covariance of projected source centers; mixture includes projected source Sigma+(.015m)^2 I plus center covariance',
            },
            'total_nodes':{k:len(a['xyz']) for k,a in assets.items()},
            'generated_color_initialization':{'rows':len(source['student_row_ids']),
              'fallback_count':int(source['fallback'].sum()),'floor_sigma_m':.015,
              'max_abs_DC_weighted_reconstruction_error':float(np.abs(reconstructed-init['coefficients'][source['student_row_ids'],0]).max()),
              'configuration':'nearest16 opacity>=.1, Euclidean<=.12m, mahalanobis^2<=9; fallback exact nearest source'},
            'views':{}}
    result['source_files'].update({name:{'path':str(path.resolve()),'sha256':sha(path)} for name,path in {
        'camera_specs':ROOT/'assets_object_clean_color_r1/camera_specs.json',
        'recorded_DC_sources':ROOT/'assets_object_clean_color_r1/generated_DC_source.npz',
        'native_teacher_front00_cache':ROOT/'teacher_cache_object_r1/novel_00.npz',
        'native_teacher_rear03_cache':ROOT/'teacher_cache_object_r1/novel_03.npz',
        'initial_native_render_receipt':ROOT/'initial_preview_object_r1/receipt.json',
        'geometry40_native_render_receipt':ROOT/'geometry_object_r1/receipt.json',
    }.items()})
    result['renderer_lowpass_source']={
        'path':'/root/autodl-tmp/drivingeditor_repro/deps/diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
        'sha256':'732e8cec40d00e93ce9959428c8e451148c791c0eaea351b8eeff9db759d7a8d',
        'line110':'cov[0][0] += 0.3f;', 'line111':'cov[1][1] += 0.3f;',
        'added_variance_px2':.3,'sigma_floor_px':float(np.sqrt(.3)),
        'actual_installed_extension':'drivingeditor diff_gaussian_rasterization/_C.cpython-39-x86_64-linux-gnu.so',
        'installed_direct_url':'file:///root/autodl-tmp/drivingeditor_repro/deps/diff-gaussian-rasterization',
        'computeCov2D_definition_line':74,'computeCov2D_call_line':217,
        'neg_fov_false_disables_lowpass':False,
    }
    adapted=load(ROOT/'assets_object_clean_color_r1/teacher_actor.npz')
    result['cached_teacher_adapter_geometry_exact']={k:bool(np.array_equal(teacher[k],adapted['rotation' if k=='rotation_wxyz' else k])) for k in ['xyz','scales','rotation_wxyz']}
    initial_receipt=json.loads((ROOT/'initial_preview_object_r1/receipt.json').read_text())
    result['initial_native_render_evidence']={'training_updates':initial_receipt['training_updates'],
        'optimizer_steps':initial_receipt['optimizer_steps'],
        'front_rear_metrics':[x for x in initial_receipt['metrics'] if x['key'] in ['novel_00','novel_03']],
        'actual_viewed_contact_sheets':['teacher_initial_native512_review.png','teacher_geometry40_native512_review.png']}
    for spec in [specs[0],specs[3]]:
        key=spec['key'];cache=load(ROOT/'teacher_cache_object_r1'/(key+'.npz'))
        domain=int((cache['alpha']>=.9).sum())
        view={'semantic_view':'front00' if key=='novel_00' else 'rear03','camera_spec':spec,
              'teacher_high_alpha_pixels':domain,'assets':{}}
        allproj={}
        for label,a in assets.items():
            uv,z,C,A,near=projected(a,covs[label],spec,cache);allproj[label]=(uv,z,C,A,near)
            selected=near&(a['opacity'].reshape(-1)>=.1)
            indices=np.flatnonzero(selected);count=len(indices)
            s0=sigmas(C[selected]);s3=sigmas(C[selected]+np.eye(2)[None]*.3)
            bins=np.rint(uv[selected]).astype(int);flat=bins[:,1]*512+bins[:,0]
            _,multiplicity=np.unique(flat,return_counts=True)
            nn=cKDTree(uv[selected]).query(uv[selected],k=2,workers=2)[0][:,1]
            part={'in_view_near_surface_rows_all_opacity':int(near.sum()),'qualified_rows':count,
                  'opacity_minimum':.1,'without_renderer_lowpass':{'major_sigma_px':stats(s0[:,0]),'minor_sigma_px':stats(s0[:,1])},
                  'with_0p3px2_renderer_lowpass':{'major_sigma_px':stats(s3[:,0]),'minor_sigma_px':stats(s3[:,1])},
                  'centers_per_1000_foreground_pixels':count/max(domain,1)*1000,
                  'unique_center_pixels':len(multiplicity),'center_pixel_multiplicity':stats(multiplicity,'count'),
                  'nearest_other_projected_center_distance_px':stats(nn),
                  'world_scale_minmax_m':[float(a['scales'].min()),float(a['scales'].max())]}
            if 'is_opaque_anchor' in a:
                part['opaque_anchor_rows_selected']=int((selected&a['is_opaque_anchor']).sum())
            view['assets'][label]=part
        uvT,zT,CT,AT,nearT=allproj['teacher']
        uvS,zS,CS,AS,nearS=allproj['student_initial']
        rows=source['student_row_ids'];idx=source['teacher_row_ids'];w=source['weights'].astype(float)
        chosen=nearS[rows]&(init['opacity'][rows]>=.1)
        uvN=uvT[idx];mu=(uvN*w[:,:,None]).sum(1);delta=uvN-mu[:,None]
        center_cov=np.einsum('nk,nki,nkj->nij',w,delta,delta)
        floorC=CT+np.einsum('nik,njk->nij',AT,AT)*(.015**2)
        mixture=center_cov+np.einsum('nk,nkij->nij',w,floorC[idx])
        c_sigma=sigmas(center_cov[chosen]);mix_sigma=sigmas(mixture[chosen])
        radius=np.sqrt((w*(delta*delta).sum(2)).sum(1))
        view['initial_generated_color_neighborhood']={
            'visible_missing_student_rows':int(chosen.sum()),
            'effective_sources_1_over_sum_w2':stats(1/(w[chosen]**2).sum(1),'count'),
            'fallback_fraction':float(source['fallback'][chosen].mean()),
            'projected_center_mix_major_sigma_px':stats(c_sigma[:,0]),
            'projected_center_mix_minor_sigma_px':stats(c_sigma[:,1]),
            'projected_source_center_RMS_radius_px':stats(radius[chosen]),
            'mixture_including_teacher_covariance_and_15mm_floor_major_sigma_px':stats(mix_sigma[:,0]),
            'mixture_including_teacher_covariance_and_15mm_floor_minor_sigma_px':stats(mix_sigma[:,1]),
            'mean_weight_from_sources_outside_same_visible_near_surface_proxy':float((w*(~nearT[idx])).sum(1)[chosen].mean()),
            'recorded_source_euclidean_distance_m':stats(source['euclidean_distance_m'][chosen][w[chosen]>0],'m'),
        }
        view['initial_to_40_max_DC_change']=float(np.abs(init['coefficients'][:,0]-assets['student_geometry40']['coefficients'][:,0]).max())
        result['views'][key]=view
    result['wall_seconds']=time.time()-started
    (OUT/'detail_footprint.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total_nodes':result['total_nodes'],'views':{k:{'sigmas':{a:{'count':x['qualified_rows'],'major_with_lpf':x['with_0p3px2_renderer_lowpass']['major_sigma_px'],'minor_with_lpf':x['with_0p3px2_renderer_lowpass']['minor_sigma_px']} for a,x in v['assets'].items()},'color':v['initial_generated_color_neighborhood']} for k,v in result['views'].items()},'seconds':result['wall_seconds']},indent=2))

if __name__=='__main__':main()
