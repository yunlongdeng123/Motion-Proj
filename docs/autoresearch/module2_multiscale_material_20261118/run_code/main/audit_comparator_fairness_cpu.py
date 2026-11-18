"""CPU只读核对matched RGB与共享场：同ci0/trace/source/coverage；不调用native。"""
from pathlib import Path
import argparse,hashlib,json,random
import numpy as np
import torch

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
FRAMES=(86,110,134,179,198)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=P/'qa/comparator_fairness.json');args=parser.parse_args()
    torch.set_num_threads(1);checks={};details={};pending=[];states={}
    schedule=[];sampler=random.Random(101)
    for _ in range(400):
        pair=[86,110];sampler.shuffle(pair);schedule.extend(pair)
    for name in('A200','L800','B500','ABC800'):
        path=P/'formal_r1'/name/'checkpoint.pt'
        if not(P/'formal_r1'/name/'stage_complete.json').exists():pending.append(name);continue
        state=torch.load(path,map_location='cpu',weights_only=False,mmap=True);states[name]=state
        checks[name+':schedule800_exact']=state['schedule']==schedule
        checks[name+':balanced_prefix_trace']=state['trace']==schedule[:state['cursor']]
        details[name]=dict(cursor=state['cursor'],trace_counts={str(k):state['trace'].count(k)for k in(86,110)},role='stage diagnostic'if name in('A200','B500')else'matched800 method')
    control=P/'matched_I0/formal_r1'
    if not(control/'completed.json').exists():pending.append('matched_I0')
    else:
        state=torch.load(control/'checkpoint_step800.pt',map_location='cpu',weights_only=False,mmap=True);states['matched_I0']=state
        checks['I0:formal800']=state['purpose']=='formal'and state['cursor']==800 and state['actual_updates']==800
        checks['I0:schedule800_exact']=state['schedule']==schedule and state['trace']==schedule
        model=state['model'];delta=model['delta'];group=state['optimizer']['param_groups']
        checks['I0:single_delta_parameter']=len(group)==1 and group[0]['params']==[0]and delta.shape==(84266,3)
        checks['I0:Adam_contract']=group[0]['lr']==.01 and tuple(group[0]['betas'])==(.9,.999)and group[0]['eps']==1e-8 and group[0]['weight_decay']==0
        adam=state['optimizer']['state'];checks['I0:Adam_step800']=set(adam)=={0}and float(adam[0]['step'])==800
        checks['I0:Adam_moments_finite_shape']=all(adam[0][k].shape==delta.shape and bool(torch.isfinite(adam[0][k]).all())for k in('exp_avg','exp_avg_sq'))
        checks['I0:raw_delta_original_box']=bool(torch.isfinite(delta).all()and(delta>=-1).all()and(delta<=1).all())
        with np.load(P/'matched_I0/input_r1/initialization.npz')as z:
            checks['I0:same_scientific_scope']=np.array_equal(state['model']['eligible'].numpy(),z['eligible'])
            difference=np.abs(state['model']['c0'].numpy().astype(np.float64)-z['c0'].astype(np.float64))
            details['I0_ci0_CPU_max']=float(difference.max());checks['I0:ci0_same_zero_field_tolerance']=float(difference.max())<=1e-6
        parent=Path(json.loads((P/'matched_I0/execution_contract.json').read_text())['coverage_parent'])
        def bytes_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
        with np.load(P/'method_scope_r1/scope.npz')as scope,np.load(parent/'road.npz')as old,np.load(control/'asset/road.npz')as new:
            for key in old.files:
                if key!='reference_srgb':checks['I0:frozen_road:'+key]=bytes_equal(old[key],new[key])
            rows=scope['active_world_rows'];eligible=scope['eligible'];world=np.zeros(len(old['active']),bool);world[rows]=eligible
            checks['I0:outside_scope_RGB_exact']=bytes_equal(old['reference_srgb'][~world],new['reference_srgb'][~world])
            expected=model['c0'].numpy().copy();ids=model['ids'].numpy();expected[ids]=np.clip(expected[ids]+delta.numpy(),0,1)
            checks['I0:saved_delta_c0_rebuilds_shader_RGB_exact']=bytes_equal(expected,new['reference_srgb'][rows])
            checks['I0:initial_readonly_RGB_exact']=bytes_equal(model['c0'].numpy()[~eligible],old['reference_srgb'][rows][~eligible])
            details['I0_actual_changed_world_rows']=int(np.any(new['reference_srgb']!=old['reference_srgb'],axis=1).sum())
        with np.load(parent/'provenance.npz')as old,np.load(control/'asset/provenance.npz')as new:
            for key in old.files:
                if key!='effective_row_rgb':checks['I0:frozen_source_ledger:'+key]=key in new.files and bytes_equal(old[key],new[key])
        for path,value in state['binding'].items():checks['I0:input_binding:'+path]=digest(path)==value
        before=[]
        for raw in FRAMES:
            with np.load(P/f'formal_r1/filtered_A0/{raw:03d}_native.npz')as base,np.load(control/f'rendered_before/{raw:03d}_native.npz')as actual:
                road=float(np.abs(base['road_rgb'].astype(np.float64)-actual['road_rgb']).max())
                joint=float(np.abs(base['joint_rgb'].astype(np.float64)-actual['joint_rgb']).max())
                alpha=np.array_equal(base['road_alpha'],actual['road_alpha'])
                checks[f'I0:zero_initial_native:{raw}']=road<=1e-6 and joint<=1e-6 and alpha
                before.append(dict(raw=raw,road_RGB_max=road,joint_RGB_max=joint,alpha_exact=alpha))
        details['I0_initial_native_vs_same_filtered_A0']=before
        details['matched_I0']=dict(cursor=state['cursor'],trace_counts={str(k):state['trace'].count(k)for k in(86,110)},
            optimizer='Adam lr.01 betas.9/.999 eps1e-8, only independentdelta',regularizer='effective row RGB delta squared *.001')
    target_paths=[P/f'stage0_native_r1/{raw:03d}_targets.npz'for raw in(86,110)]
    for name,state in states.items():
        for path in target_paths:checks[f'{name}:same_target_SHA:{path.name}']=state['binding'].get(str(path))==digest(path)
    if 'L800'in states and'ABC800'in states:
        checks['field_branches:same800_sampler_trace']=states['L800']['trace']==states['ABC800']['trace']==schedule
    if 'matched_I0'in states:
        for name in('L800','ABC800'):
            if name in states:checks[name+':same_trace_as_matchedI0']=states[name]['trace']==states['matched_I0']['trace']
    result=dict(schema='shared_material_comparator_fairness_CPU.v1',GPU_executed=False,native_calls=0,optimizer_updates=0,
        checks=checks,pending=pending,details=details,
        overall='failed'if not all(checks.values())else'incomplete'if pending else'pass',
        scientific_roles=dict(ABC800='shared low/high stages,800effective updates',L800='shared low-only800, commonA200 fork',
            matched_I0='matched same84266/filteredA0/A-C-S3131/q1/trace800, independentcolor control',
            historical_I0='different old2222 supervision, historicalonly',TRELLIS2='zero-shot prior conditionedonA/C/S, no matched-training claim'),
        non_single_variable_boundary='shared L/H regularizers and architecture differ from I0 effective-delta squared.001; compare fixed source/sampling/budget, do not call pureone-variable ablation',
        no_blind_test='two train-source views86/110, three diagnostic134/179/198, inherited199BUILD',
        human_verdict=None,production_promoted=False)
    write(args.output,result);print(json.dumps(dict(output=str(args.output),overall=result['overall'],checks=len(checks),pending=pending)))


if __name__=='__main__':main()
