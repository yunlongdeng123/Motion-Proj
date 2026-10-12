"""Exact PVG static temporal variation, CPU-only checkpoint analysis."""
from audit_checkpoint_branches_cpu import *


def inspect_time(path):
    cp,it=torch.load(path,map_location='cpu')
    xyz=cp[1].detach().numpy()
    velocity=cp[9].detach().numpy()
    sigma=np.exp(cp[8].detach().numpy()[:,0])
    time=cp[7].detach().numpy()[:,0]
    opa=torch.sigmoid(cp[6]).detach().numpy()[:,0]
    a=2*np.pi/cp[16]
    sinmin=np.ones(len(xyz),np.float32);sinmax=-sinmin.copy()
    margmin=np.ones(len(xyz),np.float32);margmax=np.zeros(len(xyz),np.float32)
    times=np.linspace(-.48,.48,49)
    for t in times:
        sin=np.sin((t-time)*a)
        mar=np.exp(-.5*((t-time)/sigma)**2)
        np.minimum(sinmin,sin,out=sinmin);np.maximum(sinmax,sin,out=sinmax)
        np.minimum(margmin,mar,out=margmin);np.maximum(margmax,mar,out=margmax)
    displacement=(sinmax-sinmin)*np.linalg.norm(velocity,axis=1)/a/model_scale
    alpha_delta=opa*(margmax-margmin)
    alpha_relative_delta=(margmax-margmin)/np.maximum(margmax,1e-30)
    nominal_anchor=sg.transform_points(xyz,model_to_anchor)
    bounds=np.array(meta['regions_anchor'][0])
    region=np.all((nominal_anchor>=bounds[0])&(nominal_anchor<=bounds[1]),axis=1)
    stage=json.loads((OUT/f'branch_static_{it}_cpu.json').read_text())
    groups=[]
    for g in stage['largest_exact_xyz_groups']:
        ids=np.flatnonzero(np.all(xyz==np.asarray(g['xyz_model'],np.float32),axis=1))
        parent=ids[0]
        live_mean=xyz[parent]+velocity[parent]*np.sin((timestamp-time[parent])*a)/a
        sh=torch.cat([cp[2][parent:parent+1],cp[3][parent:parent+1]],dim=1).detach().numpy().transpose(0,2,1)
        dirs=live_mean[None]-camera_center_model
        dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
        color=np.maximum(shutils.eval_sh(int(cp[0]),sh,dirs)+.5,0)[0]
        groups.append({'members':len(ids),'xyz_model':g['xyz_model'],
          'representative_time_center':float(time[parent]),'representative_time_sigma':float(sigma[parent]),
          'base_opacity_minmax':[float(opa[ids].min()),float(opa[ids].max())],
          'raw10_effective_opacity_minmax':[float((opa[ids]*np.exp(-.5*((timestamp-time[ids])/sigma[ids])**2)).min()),
                                           float((opa[ids]*np.exp(-.5*((timestamp-time[ids])/sigma[ids])**2)).max())],
          'max49frame_effective_opacity_quantiles':qsummary((opa*margmax)[ids]),
          'representative_raw10_SH_RGB':color.tolist(),
          'center_excursion_49frames_quantiles_m':qsummary(displacement[ids])})
    subsets={}
    for label,selected in [('all',np.ones(len(xyz),bool)),('SGCM_initial_region',region)]:
        subsets[label]={'count':int(selected.sum()),
          'center_excursion_quantiles_m':qsummary(displacement[selected]),
          'centers_excursion_gt_005m':int(((displacement>.05)&selected).sum()),
          'centers_excursion_gt_010m':int(((displacement>.1)&selected).sum()),
          'centers_excursion_gt_025m':int(((displacement>.25)&selected).sum()),
          'temporal_marginal_range_quantiles':qsummary((margmax-margmin)[selected]),
          'effective_alpha_range_quantiles':qsummary(alpha_delta[selected]),
          'relative_temporal_alpha_range_quantiles':qsummary(alpha_relative_delta[selected]),
          'mask_threshold_005_toggles':int(((margmin<=.05)&(margmax>.05)&selected).sum()),
          'never_temporally_enabled_in49frames':int(((margmax<=.05)&selected).sum()),
          'temporal_alpha_relative_range_gt_50pct':int(((alpha_relative_delta>.5)&selected).sum())}
    result={'iteration':it,'time_range':[-.48,.48],'49_actual_camera_times':True,
            'equations':'mean(t)=xyz+v*sin((t-tau)*2pi/cycle)/(2pi/cycle); alpha(t)=sigmoid(opacity)*exp(-.5*((t-tau)/exp(scaling_t))^2)',
            'static_time_invariance_consistency':'High-risk gap: Eq4 prose calls static background time-invariant. Source still trains/renders PVG time, sigma, velocity. Figure3 discusses suppression of static v, so do not silently zero parameters or claim author exact implementation.',
            'subsets':subsets,'largest_exact_xyz_groups':groups,
            'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized()}
    save(f'static_time_{it}_cpu.json',result)
    print(json.dumps({'iteration':it,'time_variation':subsets['all'],'largest_group_raw10_effective_alpha':groups[0]['raw10_effective_opacity_minmax']},ensure_ascii=False),flush=True)
    del cp
    gc.collect()
    return result


if __name__=='__main__':
    r=[inspect_time(RUN/'chkpnt20000.pth'),inspect_time(RUN/'chkpnt30000.pth')]
    save('static_time_summary.json',r)
    # Independent numerical plane/metric and stored time-normal cross-check.
    world,t=read_ply(DATA/'points3d.ply')
    mod=sg.transform_points(world,transform)
    pmod=np.asarray(meta['plane_model'])
    anchor=sg.transform_points(world,world_to_anchor)
    err=np.abs(np.abs(mod@pmod[:3]+pmod[3])/model_scale-np.abs(anchor@plane[:3]+plane[3]))
    checks={}
    for name in ['points3d.ply','points3d_dynamic_0.ply']:
        v=PlyData.read(str(DATA/name))['vertex']
        checks[name]={'nx_ny_nz_exact_equal':bool(np.array_equal(v['nx'],v['ny']) and np.array_equal(v['nx'],v['nz'])),
                      'fields':list(v.data.dtype.names),'normal_fields_semantics':'All3normal fields encode model timestamp, not physical surface normal.'}
    save('metric_and_normal_invariants.json',{'max_perpendicular_distance_error_m':float(err.max()),
                  'metric_distance_invariance_pass':bool(err.max()<1e-7),'PLY_time_normal_checks':checks,'gpu_used':False,
                  'model_transform_similarity_scale':sg.validate_similarity(transform)})
    print('STATIC_TIME_CPU_AUDIT_COMPLETE',flush=True)
