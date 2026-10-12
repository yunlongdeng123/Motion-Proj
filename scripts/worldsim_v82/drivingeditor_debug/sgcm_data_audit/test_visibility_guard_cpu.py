"""Adversarial CPU regression and checkpoint dead-clone eligibility evidence."""
from audit_cpu import *
candidate=load_module(OUT/'candidate/paper_sgcm_visibility_guard.py','sgcm_visibility_candidate')


class TorchCPUAdapter:
    def __getattr__(self,name):return getattr(torch,name)
    def zeros(self,*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return torch.zeros(*args,**kwargs)


ns=dict(torch=TorchCPUAdapter(),nn=nn,np=np,inverse_sigmoid=lambda x:torch.log(x/(1-x)))
methods=['replace_tensor_to_optimizer','_prune_optimizer','prune_points','cat_tensors_to_optimizer','densification_postfix']
exec(ast_functions(SRC/'scene/gaussian_model.py','GaussianModel',methods),ns)
class Harness:
    get_xyz=property(lambda s:s._xyz)
    get_scaling=property(lambda s:torch.exp(s._scaling))
    get_opacity=property(lambda s:torch.sigmoid(s._opacity))
    get_t=property(lambda s:s._t)
    get_scaling_t=property(lambda s:torch.exp(s._scaling_t))
    scaling_inverse_activation=staticmethod(torch.log)
for name in methods:setattr(Harness,name,ns[name])


def model(height=.3,opacity=.1,time=0.,sigma=.31,quat=None):
    h=Harness();h.time_duration=[-.48,.48]
    fields=['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_t','_scaling_t','_velocity']
    names=['xyz','f_dc','f_rest','opacity','scaling','rotation','t','scaling_t','velocity']
    q=[1.,0,0,0] if quat is None else quat
    values=[[[0,0,height],[10,0,.1]],[[[10]],[[20]]],np.arange(12).reshape(2,2,3),
            [[np.log(opacity/(1-opacity))],[0]],[[np.log(.1),np.log(.2),np.log(.3)]]*2,
            [q,[1.,0,0,0]],[[time],[0]],[[np.log(sigma)],[np.log(.31)]],[[0,0,0],[0,0,0]]]
    for f,v in zip(fields,values):setattr(h,f,nn.Parameter(torch.tensor(v,dtype=torch.float64)))
    h.optimizer=torch.optim.Adam([{'params':[getattr(h,f)],'name':n,'lr':.001} for f,n in zip(fields,names)])
    for name in ['max_radii2D','xyz_gradient_accum','t_gradient_accum','denom']:
        setattr(h,name,torch.zeros(2) if name=='max_radii2D' else torch.zeros(2,1))
    return h


def controller(module,guard):
    values=dict(start=0,end=1000,interval=100,offset=0,max_new_points=10000)
    if module is candidate:values['require_effective_visibility']=guard
    return module.SGCMController(module.SGCMConfig(**values),1,np.array([0.,0,1,0]),np.eye(4),
                                [np.array([[-1.,-1.,-1.],[1.,1.,1.]])],{})


def expansion(module,guard,dead_alpha=False):
    h=model(opacity=.01,time=-4.916017055511475,sigma=2.991501808166504) if dead_alpha else model()
    c=controller(module,guard);counts=[len(h.get_xyz)];stats=[]
    for it in range(100,701,100):
        radius=100. if dead_alpha else 0.
        s=c.refine(h,torch.full((len(h.get_xyz),),radius,dtype=torch.float64),it)
        counts.append(len(h.get_xyz));stats.append(s)
    return {'counts':counts,'last_stats':stats[-1]}


def regressions():
    zero_baseline=expansion(sg,False);zero_candidate=expansion(candidate,True)
    alpha_baseline=expansion(sg,False,True);alpha_candidate=expansion(candidate,True,True)
    assert zero_baseline['counts']==[2,3,5,9,17,33,65,129]
    assert zero_candidate['counts']==[2]*8
    assert alpha_baseline['counts']==[2,3,5,9,17,33,65,129]
    assert alpha_candidate['counts']==[2]*8
    # Disabling the new gate keeps the frozen behavior exactly.
    off=expansion(candidate,False)
    assert off['counts']==zero_baseline['counts']
    livea,liveb=model(),model()
    sa=controller(sg,False).refine(livea,torch.tensor([30.,0.]),100)
    sb=controller(candidate,True).refine(liveb,torch.tensor([30.,0.]),100)
    assert len(livea.get_xyz)==len(liveb.get_xyz)==3
    assert all(torch.equal(getattr(livea,f),getattr(liveb,f)) for f in ['_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation','_t','_scaling_t','_velocity'])
    p=model(height=.8,opacity=.01,time=-5,sigma=.1)
    sp=controller(candidate,True).refine(p,torch.tensor([0.,0.]),100)
    assert len(p.get_xyz)==1 and sp['pruned']==1
    # Nontrivial wxyz rotation: independent SciPy covariance/axis expectation.
    from scipy.spatial.transform import Rotation
    xyzw=Rotation.from_euler('xyz',[35,70,-20],degrees=True).as_quat()
    q=[xyzw[3],*xyzw[:3]]
    s=model(height=.1,quat=q)
    ss=controller(candidate,True).refine(s,torch.tensor([30.,0.]),100)
    children=s.get_xyz.detach().numpy()[1:]
    matrix=Rotation.from_quat(xyzw).as_matrix()
    offset=matrix[:,2]*(.3/np.sqrt(2))
    assert np.allclose(children,np.array([[0,0,.1]-offset,[0,0,.1]+offset]),atol=1e-12)
    childsig=np.array([.1,.2,.3/np.sqrt(2)])
    cov_child=matrix@np.diag(childsig**2)@matrix.T
    recovered_cov=cov_child+np.outer(offset,offset)
    expected_cov=matrix@np.diag(np.array([.1,.2,.3])**2)@matrix.T
    assert np.allclose(recovered_cov,expected_cov,atol=1e-12)
    assert np.array_equal(s._features_dc.detach().numpy()[1:],[[[10]],[[10]]])
    upper=candidate.max_effective_opacity_numpy([.01,.01],[0,-4.9160170555],[1,2.9915018082],[-.48,.48])
    assert upper[0]==.01 and upper[1]<candidate.RASTERIZER_ALPHA_CUTOFF
    result={'tests_passed':7,'radii_zero_baseline':zero_baseline,'radii_zero_candidate':zero_candidate,
            'positive_radii_globally_under_alpha_baseline':alpha_baseline,
            'positive_radii_globally_under_alpha_candidate':alpha_candidate,
            'default_disabled_matches_baseline':True,'eligible_live_clone_math_exactly_unchanged':True,
            'geometric_pruning_unchanged_even_for_dead_alpha':True,
            'nontrivial_quaternion_split_axis_and_mixture_covariance_pass':True,
            'guard_is_declared_unreported_eligibility_choice':True,
            'no_ground_threshold_or_loss_change':True,'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized()}
    save('visibility_guard_cpu_regression.json',result)
    print('VISIBILITY_GUARD_REGRESSION_PASS',flush=True)


def affected_rows():
    meta=json.loads((OUT/'sgcm_runtime_geometry.json').read_text())
    transform=np.asarray(meta['world_to_model'])
    pose=np.loadtxt(DATA/'pose/0017000.txt')
    model_to_anchor=np.linalg.inv(pose)@np.linalg.inv(transform)
    plane=np.asarray(meta['plane_model']);scale=meta['meter_to_model_scale']
    bounds=np.asarray(meta['regions_anchor'][0])
    rows=[]
    for path in [BASE/'runs/scene0017_000_048_paper_50000_seed0/chkpnt10000.pth',RUN/'chkpnt20000.pth',RUN/'chkpnt30000.pth']:
        cp,it=torch.load(path,map_location='cpu')
        xyz=cp[1].detach().numpy();time=cp[7].detach().numpy()[:,0];sigma=np.exp(cp[8].detach().numpy()[:,0])
        opa=torch.sigmoid(cp[6]).detach().numpy()[:,0]
        upper=candidate.max_effective_opacity_numpy(opa,time,sigma,[-.48,.48])
        extended=candidate.max_effective_opacity_numpy(opa,time,sigma,[-.51,.51])
        anchor=sg.transform_points(xyz,model_to_anchor)
        distance=np.abs(xyz@plane[:3]+plane[3])/scale
        inside=np.all((anchor>=bounds[0])&(anchor<=bounds[1]),axis=1)
        clone=inside&(distance>.25)&(distance<=.5)
        dead=upper<candidate.RASTERIZER_ALPHA_CUTOFF
        storedzero=cp[10].detach().numpy()<=0
        row={'iteration':it,'static_points':len(xyz),'geometric_clone_candidates':int(clone.sum()),
             'always_under_alpha_all_static':int(dead.sum()),
             'always_under_alpha_geometric_clone_candidates':int((dead&clone).sum()),
             'always_under_alpha_extended_jitter_geometric_clone_candidates':int(((extended<candidate.RASTERIZER_ALPHA_CUTOFF)&clone).sum()),
             'checkpoint_stored_zero_radius_clone_candidates':int((storedzero&clone).sum()),
             'radii_note':'checkpoint max_radii2D is historical and must not be equated to current rendered radii; no training-time affected count is inferred from it.',
             'actual_static_time_interval':[-.48,.48],
             'hypothetical_jitter_time_interval':[-.51,.51],
             'static_jitter_behavior':'Frozen prepare_rasterization else branch assigns time_shift=None. Training samples shift in +/-0.03 model time, but it affects dynamic branch only.'}
        if it==20000:
            groups=json.loads((OUT/'branch_static_20000_cpu.json').read_text())['largest_exact_xyz_groups']
            arrays={'always_under_alpha_clone_indices':np.flatnonzero(dead&clone)}
            records=[]
            for i,g in enumerate(groups):
                ids=np.flatnonzero(np.all(xyz==np.asarray(g['xyz_model'],dtype=np.float32),axis=1))
                arrays[f'group{i}_indices']=ids
                first=int(ids[0]);world=sg.transform_points(xyz[first:first+1],np.linalg.inv(transform))[0]
                records.append({'npz_key':f'group{i}_indices','count':len(ids),'first_index':first,
                    'model_xyz':xyz[first].tolist(),'world_xyz':world.tolist(),
                    'time_center_quantiles':qsummary(time[ids]),'time_sigma_quantiles':qsummary(sigma[ids]),
                    'continuous_alpha_upper_bound_quantiles':qsummary(upper[ids]),
                    'extended_jitter_alpha_upper_bound_quantiles':qsummary(extended[ids]),
                    'all_continuous_upper_bounds_under_cutoff':bool((upper[ids]<candidate.RASTERIZER_ALPHA_CUTOFF).all())})
            file=OUT/'static20k_dead_clone_and_duplicate_indices.npz'
            np.savez_compressed(file,**arrays)
            save('static20k_dead_clone_index_manifest.json',{'checkpoint':str(path),
                 'index_semantics':'zero-based frozen checkpoint Gaussian row index; independent copies; no formal pruning performed',
                 'index_npz':str(file),'index_npz_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                 'renderer_alpha_cutoff':candidate.RASTERIZER_ALPHA_CUTOFF,'continuous_bound_formula':'opacity*exp(-.5*((time_center-clip(time_center,tmin,tmax))/sigma)^2)',
                 'time_interval':[-.48,.48],'extended_hypothetical_jitter_interval':[-.51,.51],
                 'always_under_alpha_clone_count':int((dead&clone).sum()),'groups':records,'gpu_used':False})
        rows.append(row)
        del cp
        gc.collect()
    save('visibility_guard_affected_rows.json',{'rows':rows,'candidate_status':'Isolated, disabled by default, not deployed; no GPU or training executed',
        'does_not_repair_preexisting_dead_clones_or_prove_confetti_causality':True,
        'gpu_used':False,'cuda_initialized':torch.cuda.is_initialized()})
    print(json.dumps(rows,ensure_ascii=False),flush=True)


if __name__=='__main__':
    regressions()
    affected_rows()
    assert not torch.cuda.is_initialized()
    print('VISIBILITY_GUARD_CPU_AUDIT_COMPLETE',flush=True)
