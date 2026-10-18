"""Read-only native CUDA checks across the 3 -> 5 identity batch boundary.

No optimizer updates and no replacement rasterizer. A single-ID request still
contains every scene occluder, providing an independent feature-packing route.
"""
from pathlib import Path
import os,sys,json,time,hashlib,ast,math,traceback
from types import SimpleNamespace
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
TASK=ROOT/'identity_completion_20261018'
OUT=ROOT/'baseline_engineering_20261018'
sys.path.insert(0,str(TASK/'infra'))
from priority import require_gpu_slot

def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False))

def compare(a,b,atol=3e-6,rtol=2e-4):
    import torch
    delta=(a-b).abs();ref=b.abs()
    return dict(max_abs=float(delta.max()),relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref).clamp_min(1e-20)),
                passed=bool(torch.all(delta<=atol+rtol*ref)),atol=atol,rtol=rtol)

def tiny(renderer,torch):
    import torch.nn.functional as F
    fixture=ast.parse((OUT/'source/test_renderer_contract.py').read_text())
    names={'camera','pipe','StaticPVG','WorldState','Instances'}
    ns=dict(torch=torch,F=F,math=math,SimpleNamespace=SimpleNamespace)
    exec(compile(ast.Module(body=[n for n in fixture.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),'fixtures','exec'),ns)
    s=ns['StaticPVG'](depth=1.,opacity=.2)
    m=ns['Instances'](depths=(2.,3.,4.,5.,6.),opacities=(.2,.25,.3,.35,.4))
    for obj in (s,m.state):
        for k,v in vars(obj).copy().items():
            if torch.is_tensor(v):setattr(obj,k,v.detach().cuda().requires_grad_(v.requires_grad))
    m.state.xyz.requires_grad_(True);m.state.scaling.requires_grad_(True)
    s.get_xyz_SHM=lambda t:s.get_xyz
    def render(ids,max_features):
        return renderer.render(ns['camera'](device='cuda'),s,m,ns['pipe'](),torch.zeros(3,device='cuda'),
            other=torch.zeros(1,4,device='cuda'),instance_ids=ids,max_features=max_features,render_separate=False)
    ids=list(m.instance_ids)
    a=render(ids,10);b=render(ids,8);rev=render(ids[::-1],10)
    tests=dict(batch2_vs_single_passes=compare(a['instance_visible_contribution'],b['instance_visible_contribution']),
        permuted_slots=compare(a['instance_visible_contribution'],rev['instance_visible_contribution'].flip(0)),
        sum_equals_visible_union=compare(a['instance_visible_contribution'].sum(0,keepdim=True),a['dynamic_visible_alpha']))
    params=(m.state.opacity,m.state.xyz,m.state.scaling,s.get_opacity)
    weight=torch.arange(1,6,device='cuda')[:,None,None]
    grads=[]
    for pack,w in [(a,weight),(b,weight),(rev,weight.flip(0))]:
        grads.append(torch.autograd.grad((pack['instance_visible_contribution']*w).sum(),params))
    for route,g in zip(['one_ID_per_pass','permuted_slots'],grads[1:]):
        for name,x,y in zip(['actor_opacity','actor_xyz','actor_scale','static_opacity'],grads[0],g):
            tests[route+'_gradient_'+name]=compare(x,y,atol=3e-6,rtol=5e-4)
    nonzero_per_identity=grads[0][0].abs().flatten().tolist()
    assert all(v>0 for v in nonzero_per_identity)
    return dict(tests=tests,actor_opacity_gradient_abs=nonzero_per_identity,
        passed=all(x['passed'] for x in tests.values()),passes=[a['instance_feature_passes'],b['instance_feature_passes']])

def scene(renderer,torch):
    from runtime import load_models,masks_for_frame
    ck=TASK/'train5/checkpoint_030000.pt'
    before=(ck.stat().st_size,ck.stat().st_mtime_ns)
    saved=torch.load(ck,map_location='cuda',weights_only=False)
    spec=json.loads((TASK/'spec5.json').read_text())
    scratch=OUT/'readonly_model_load';scratch.mkdir()
    cfg,sc,s,m,opt,env,exclusion=load_models(spec,scratch,restore=saved)
    del saved
    tracks=list(m.instance_ids);tests=[];gradient_tests=[]
    params=(m._opacity,m._xyz,m._scaling,s._opacity,s._xyz)
    names=['actor_opacity','actor_xyz','actor_scale','static_opacity','static_xyz']
    for raw in [107,128,137]:
        cam=next(c for c in sc.getTrainCameras() if Path(c.image_name).stem==f'0008{raw:03d}')
        other=[s.get_scaling_t.clamp_max(2),s.get_inst_velocity]
        def render(ids):
            return renderer.render(cam,s,m,cfg,torch.zeros(3,device='cuda'),env_map=env,
                other=other,instance_ids=ids,max_features=10,render_separate=False)
        with torch.no_grad():
            a=render(tracks);r=render(tracks[::-1])
            tests.append(dict(raw=raw,name='full_RGB_independent_of_identity_slots',**compare(a['render'],r['render'])))
            tests.append(dict(raw=raw,name='permuted_identity_slots',**compare(a['instance_visible_contribution'],r['instance_visible_contribution'].flip(0))))
            tests.append(dict(raw=raw,name='sum_equals_visible_union',**compare(a['instance_visible_contribution'].sum(0,keepdim=True),a['dynamic_visible_alpha'])))
            for i,t in enumerate(tracks):
                b=render([t])
                tests.append(dict(raw=raw,name='one_ID_reference',slot=i,track=t,**compare(a['instance_visible_contribution'][i:i+1],b['instance_visible_contribution'])))
            del a,r,b
        if raw!=128:continue
        masks,known=masks_for_frame(spec,m,cam)
        for i,t in enumerate(tracks):
            a=render(tracks)
            ga=torch.autograd.grad((a['instance_visible_contribution'][i]-masks[i]).square().mean(),params,allow_unused=True)
            del a
            b=render([t])
            gb=torch.autograd.grad((b['instance_visible_contribution'][0]-masks[i]).square().mean(),params,allow_unused=True)
            for name,x,y in zip(names,ga,gb):
                if x is None or y is None:
                    gradient_tests.append(dict(slot=i,track=t,parameter=name,passed=False,error='disconnected gradient'))
                else:
                    gradient_tests.append(dict(slot=i,track=t,parameter=name,norm=float(torch.linalg.vector_norm(y)),**compare(x,y,atol=5e-6,rtol=5e-3)))
            del b,ga,gb
            write('progress.json',dict(stage='scene_gradient',raw=raw,identity_slot=i,parameter_updates=0))
    assert before==(ck.stat().st_size,ck.stat().st_mtime_ns)
    return dict(tests=tests,gradient_tests=gradient_tests,
        passed=all(x['passed'] for x in tests+gradient_tests),checkpoint_unchanged=True,
        frames=[107,128,137],checkpoint=str(ck),checkpoint_stat=before)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if (OUT/'channel_gate.json').exists():raise FileExistsError('Preserve existing gate receipt; diagnose before rerun')
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    started=time.monotonic();result={}
    try:
        import torch,renderer,diff_gaussian_rasterization
        torch.set_num_threads(2)
        result.update(admission=admission,parameter_updates=0,device=torch.cuda.get_device_name(),
            renderer_sha256=hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest(),
            native_rasterizer=str(diff_gaussian_rasterization.__file__))
        result['tiny']=tiny(renderer,torch);write('progress.json',dict(stage='tiny_done',passed=result['tiny']['passed']))
        result['scene']=scene(renderer,torch)
        result['status']='PASSED' if result['tiny']['passed'] and result['scene']['passed'] else 'FAILED'
    except BaseException:
        result['status']='FAILED';result['error']=traceback.format_exc();raise
    finally:
        result['seconds']=time.monotonic()-started;write('channel_gate.json',result);slot.close()
        print(json.dumps(result),flush=True)
