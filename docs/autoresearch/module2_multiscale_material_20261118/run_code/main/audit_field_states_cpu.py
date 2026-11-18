"""独立只读CPU审核共享场阶段：固定geometry/sourceledger、scope、Adam计数与sampler。

不导入native、不重新渲染；chart修复的独立资格引用root/static交叉核，不自我认证。
"""
from pathlib import Path
import argparse,hashlib,json,random
import numpy as np
import torch

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
N=P.parent/'road_activation_20261118_r1'
PARENT=N/'B32_beam95_native_r1/asset_guarded'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def exact(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b)and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(exact(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b)and len(a)==len(b)and all(exact(x,y)for x,y in zip(a,b))
    return a==b


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def check_stage(folder,expected,reference_A=None,reference_B=None,hash_cache=None):
    checks={};details={};folder=Path(folder);asset=folder/'asset'
    s=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    checks['role_formal']=s['purpose']=='formal'
    checks['cursor_and_logical_updates']=s['cursor']==expected['cursor']==s['actual_updates']
    schedule=[];sampler=random.Random(101)
    for _ in range(400):
        pair=[86,110];sampler.shuffle(pair);schedule.extend(pair)
    checks['full_sampler_contract']=s['schedule']==schedule
    checks['trace_prefix_exact']=s['trace']==schedule[:expected['cursor']]
    checks['only_two_shared_parameters']=set(s['params'])=={'low','high'}
    checks['parameters_finite']=all(bool(torch.isfinite(x).all())for x in s['params'].values())
    checks['stage_and_arm']=s['stage']==expected['stage']and s['arm']==expected['arm']
    groups=s['optimizer']['param_groups'];adam=s['optimizer']['state']
    checks['Adam_hyperparameters']=len(groups)==2 and [g['lr']for g in groups]==[.01,.003]and all(
        tuple(g['betas'])==(.9,.999)and g['eps']==1e-8 and g['weight_decay']==0 for g in groups)
    checks['Adam_parameter_ids']=[g['params']for g in groups]==[[0],[1]]
    for index,key in[(0,'low'),(1,'high')]:
        count=expected[key+'_steps'];a=adam.get(index)
        checks[key+'_Adam_step']=a is None if count==0 else a is not None and float(a['step'])==count
        if a is not None:
            checks[key+'_Adam_moment_shape_finite']=all(a[k].shape==s['params'][key].shape and bool(torch.isfinite(a[k]).all())for k in('exp_avg','exp_avg_sq'))
    checks['high_is_zero_when_frozen']=bool((s['params']['high']==0).all())if expected['high_steps']==0 else True
    field=torch.load(asset/'world_field.pt',map_location='cpu',weights_only=False,mmap=True)
    fs=field['state'];settings=field['settings']
    checks['serialized_parameters_exact']=all(exact(s['params'][k],fs[k])for k in('low','high'))
    checks['registered_bounds_and_spacing']=settings==dict(low_stride=6,low_bound=.3,high_bound=.04,highpass_sigma_cells=3.)
    checks['GH15_exact_chart_present']=fs['weights'].shape==(225,)and fs['sample_uv'].shape==(151988,225,2)and fs['geometric_sample_valid'].shape==(151988,225)
    checks['frozen_parameter_support']=bool((fs['low'][fs['low_parameter_mask'][...,0]==0]==0).all())and bool((fs['high'][fs['high_parameter_mask'][...,0]==0]==0).all())
    with np.load(P/'cpu_exact_r1/A0.npz')as source:
        for key,source_key in [('A0','rgb_srgb'),('valid','valid'),('mH','mH')]:
            value=source[source_key]
            if key in('valid','mH'):value=value[...,None]
            checks['serialized_source_buffer:'+key]=torch.equal(fs[key],torch.as_tensor(value,dtype=fs[key].dtype))
        gate=source['mH'];permit=(source['source_effective']==1)&source['valid']
        checks['H_mask_real_observed_source_only']=bool(np.all(gate[~permit]==0))
        checks['H_mask_finite_unit_interval']=bool(np.isfinite(gate).all()and np.all((gate>=0)&(gate<=1)))
        details['high_permission_nodes']=int((gate>0).sum())
        details['high_permission_generated_retrieved_unknown_nodes']=int(((gate>0)&~permit).sum())
    with np.load(P/'method_scope_r1/scope.npz')as z:scope={k:z[k]for k in z.files}
    with np.load(asset/'field_scope.npz')as z:
        checks['scientific_scope_exact']=exact(z['active_world_rows'],scope['active_world_rows'])and exact(z['eligible'],scope['eligible'])
    checks['common_write_rows_84266']=int(scope['eligible'].sum())==84266
    with np.load(PARENT/'road.npz')as parent,np.load(asset/'road.npz')as road:
        checks['road_keys_exact']=parent.files==road.files
        for key in parent.files:
            if key!='reference_srgb':checks['road_frozen:'+key]=exact(parent[key],road[key])
        rgb=road['reference_srgb'];base=parent['reference_srgb'];full=np.zeros(len(base),bool)
        full[scope['active_world_rows']]=scope['eligible']
        checks['outside_scientific_scope_RGB_exact']=exact(base[~full],rgb[~full])
        checks['shader_RGB_finite_box']=bool(np.isfinite(rgb).all()and np.all(rgb>=0)and np.all(rgb<=1))
        details['actual_changed_world_rows']=int(np.any(base!=rgb,axis=1).sum())
        details['eligible_rows']=int(full.sum())
    # 仅以下派生capacity/effective字段可重写；原known/source/unknown/UV/observed等逐byte审核。
    derived={'capacity_source_prior_reference_srgb','capacity_effective_rgb','effective_row_rgb','capacity_color_eligible','capacity_prior_ledger_immutable','asset_mode'}
    with np.load(PARENT/'provenance.npz')as parent,np.load(asset/'provenance.npz')as new:
        for key in parent.files:
            if key not in derived:checks['source_ledger_frozen:'+key]=key in new.files and exact(parent[key],new[key])
        checks['effective_RGB_matches_shader']=exact(new['effective_row_rgb'],rgb)and exact(new['capacity_effective_rgb'],rgb)
        checks['prior_effective_RGB_explicit']=exact(new['capacity_source_prior_reference_srgb'],base)
        checks['serialized_color_scope']=exact(new['capacity_color_eligible'],full)
    cache={}if hash_cache is None else hash_cache
    for path,sha in s['binding'].items():
        p=Path(path)
        if path not in cache:cache[path]=digest(p)if p.exists()else None
        checks['source_binding:'+path]=cache[path]==sha
    receipt=read(asset/'field_receipt.json')
    checks['field_sha_binding']=digest(asset/'world_field.pt')==receipt['field_sha256']
    checks['derived_road_sha_binding']=digest(asset/'road.npz')==receipt['derived_road_sha256']
    checks['scope_sha_binding']=digest(asset/'field_scope.npz')==receipt['field_scope_sha256']
    checks['public_persistent_scene_and_proof_present']=(asset/'persistent/scene.json').exists()and(asset/'persistent/proof.json').exists()
    if reference_A is not None:
        A=torch.load(Path(reference_A)/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        if expected['stage']=='B':
            checks['B_low_parameter_frozen_exact']=exact(s['params']['low'],A['params']['low'])
            checks['B_low_Adam_frozen_exact']=exact(adam[0],A['optimizer']['state'][0])
        details['low_raw_drift_from_A200_max']=float((s['params']['low']-A['params']['low']).abs().max())
    if reference_B is not None:
        B=torch.load(Path(reference_B)/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
        details['high_raw_drift_from_B500_max']=float((s['params']['high']-B['params']['high']).abs().max())
    details.update(logical_updates=s['cursor'],trace_counts={str(raw):s['trace'].count(raw)for raw in(86,110)},
        low_effective_abs_max=float((.3*torch.tanh(s['params']['low'])).abs().max()),
        raw_high_abs_max=float(s['params']['high'].abs().max()),
        low_open_latent_nodes=int((fs['low_parameter_mask']>0).sum()),
        high_open_latent_nodes=int((fs['high_parameter_mask']>0).sum()),
        high_output_support_nodes=int((fs['high_output_mask']>0).sum()),
        low_open_scalar_parameters=3*int((fs['low_parameter_mask']>0).sum()),
        high_open_scalar_parameters=3*int((fs['high_parameter_mask']>0).sum()),
        parameter_count_boundary='open shared latent coefficients, not independent ci or a claim of numericalrank',
        registered_effective_H_bound=.04,stage_statistics=read(folder/'field_statistics.json'),
        frozen_ledger_exceptions=sorted(derived),geometry_chart_authority='same root/static independent chart qualification; this script does not self-certify chart fix')
    return dict(status='pass'if all(checks.values())else'failed',checks=checks,details=details)


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=P/'qa/field_stage_acceptance.json')
    args=parser.parse_args();cache={};stages={}
    plans=[('A200',dict(cursor=200,stage='A',arm='common',low_steps=200,high_steps=0)),
        ('L800',dict(cursor=800,stage='L',arm='L_control',low_steps=800,high_steps=0)),
        ('B500',dict(cursor=500,stage='B',arm='ABC',low_steps=200,high_steps=300)),
        ('ABC800',dict(cursor=800,stage='C',arm='ABC',low_steps=500,high_steps=600))]
    missing=[]
    for name,expected in plans:
        folder=P/'formal_r1'/name
        if not(folder/'stage_complete.json').exists():missing.append(name);continue
        stages[name]=check_stage(folder,expected,reference_A=P/'formal_r1/A200'if name in('B500','ABC800')else None,
            reference_B=P/'formal_r1/B500'if name=='ABC800'else None,hash_cache=cache)
    result=dict(schema='shared_material_independent_CPU_state_audit.v1',GPU_executed=False,native_calls=0,
        new_optimizer_updates=0,stages=stages,pending=missing,
        overall='failed'if any(s['status']=='failed'for s in stages.values())else'incomplete'if missing else'pass',
        actual_native_K_alpha_and_field_reload='separate five-frame float/lifecycle receipts; no native call by this audit',
        human_verdict=None,production_promoted=False)
    write(args.output,result)
    print(json.dumps(dict(output=str(args.output),overall=result['overall'],
        stages={k:dict(status=v['status'],checks=len(v['checks']),failed=[x for x,y in v['checks'].items()if not y])for k,v in stages.items()},pending=missing)))


if __name__=='__main__':main()
