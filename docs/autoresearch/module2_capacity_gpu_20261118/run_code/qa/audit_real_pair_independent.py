"""CPU独立重算门3配对；不导入renderer，不修改正式输出。"""
from pathlib import Path
import argparse,hashlib,json,os,random
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np

FRAMES=(86,110,134,179,198)
PHOTO_ROLES=dict(A=('A_mask','q_A','target_A_srgb'),C=('C_hole_photo','q_C_hole','target_C_srgb'),S=('S_target_shadow_photo','q_S','target_S_srgb'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def byte_equal(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def premult_mae(rgb,alpha,teacher,mask,q):
    w=np.asarray(mask,bool)*np.asarray(q,np.float64)
    if w.sum()<=0:raise ValueError('empty fixed teacher domain')
    error=np.abs(np.asarray(rgb,np.float64)-np.asarray(alpha,np.float64)[None]*np.asarray(teacher,np.float64))
    return float((error*w[None]).sum()/(3*w.sum()))


def frequency_metrics(rgb,base,domain,vertical,horizontal):
    rgb=np.asarray(rgb,np.float64);base=np.asarray(base,np.float64)
    return dict(E_vertical=float(np.abs(rgb[:,2:]-2*rgb[:,1:-1]+rgb[:,:-2])[:,vertical].mean()),
        E_horizontal=float(np.abs(rgb[:,:,2:]-2*rgb[:,:,1:-1]+rgb[:,:,:-2])[:,horizontal].mean()),
        RGB_delta_energy=float(np.square(rgb-base)[:,domain].mean()))


def excess_gate(baseline,independent,shared):
    excess=float(independent-baseline)
    if excess<=0:return dict(status='NOT_APPLICABLE',I0_excess=excess,W4_excess=float(shared-baseline),ratio=None,pass_gate=None)
    ratio=float((shared-baseline)/excess)
    return dict(status='PASS' if ratio<=.5 else 'FAIL',I0_excess=excess,W4_excess=float(shared-baseline),ratio=ratio,pass_gate=ratio<=.5)


def expected_trace(steps,seed=101):
    sampler=random.Random(seed);queue=[];trace=[]
    for _ in range(steps):
        if not queue:queue=[86,110];sampler.shuffle(queue)
        trace.append(queue.pop())
    return trace,queue,sampler.getstate()


def compare_snapshot(a,b,path=''):
    """Return only differences, including complete RNG/sampler/contract state."""
    import torch
    if torch.is_tensor(a):
        if torch.is_tensor(b) and torch.equal(a,b):return []
        return [dict(path=path,type='tensor',max_abs=float((a.double()-b.double()).abs().max()))]
    if isinstance(a,np.ndarray):return [] if isinstance(b,np.ndarray) and byte_equal(a,b) else [dict(path=path,type='numpy')]
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():return [dict(path=path,type='dict_keys')]
        return [d for k in a for d in compare_snapshot(a[k],b[k],path+'/'+str(k))]
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):return [dict(path=path,type='sequence')]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in compare_snapshot(x,y,path+'/'+str(i))]
    return [] if a==b else [dict(path=path,type='scalar')]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('This independent audit is CPU only')
    ap=argparse.ArgumentParser();ap.add_argument('--gpu-campaign',type=Path,required=True);ap.add_argument('--cpu-campaign',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();g,c=args.gpu_campaign,args.cpu_campaign;pair=g/'check3';args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('preserve previous independent audit: '+str(args.output))
    folders={arm:pair/arm/'formal_r1' for arm in ['I0','W4']}
    missing=[str(path) for folder in folders.values() for path in [folder/'completed.json',folder/'public_reload.json'] if not path.exists()]
    if missing:raise RuntimeError('Inputs not complete: '+str(missing))
    import torch
    torch.set_num_threads(1)
    def checkpoint(p):return torch.load(p,map_location='cpu',weights_only=False)
    protocol=read(c/'protocol_cpu_prepared.json')['check3'];checks={};notes={};pins={}
    contracts={arm:read(folder/'frozen_contract.json') for arm,folder in folders.items()}
    common={arm:{k:v for k,v in value.items() if k!='arm'} for arm,value in contracts.items()}
    checks['same_frozen_contract_except_arm']=common['I0']==common['W4']
    for arm,value in contracts.items():
        checks[arm+'_formal_200_seed101']=value['formal'] and value['steps']==200 and value['seed']==101
        checks[arm+'_optimizer_loss_bounds_TV_match_parent']=value['optimizer']==protocol['optimizer'] and value['loss']==protocol['objective'] and value['delta_node_bounds']==protocol['delta_node_bounds'] and value['TV']==0
        for p,digest in value['input_binding'].items():
            if p not in pins:pins[p]=dict(expected=digest,actual=sha(p))
            checks['pin_'+str(len(checks))]=pins[p]['actual']==digest
    checks['both_arm_same_teacher_scope_input_bindings']=contracts['I0']['input_binding']==contracts['W4']['input_binding']
    scope=load(g/'check1/native_r1/qualified_scope.npz');eligible=scope['eligible_color'];active=scope['active_world_rows']
    expected=scope['physical_legal']&~scope['conservative_K_guard']&~scope['near_guard']&(scope['A_seen']|scope['S_seen']|(scope['C_seen']&~scope['known']))
    checks['new_V4_actual_scope_exact']=np.array_equal(expected,eligible) and int(eligible.sum())==83388
    checks['new_S_only_known_3637_included']=int(scope['S_only_known'].sum())==3637 and bool(eligible[scope['S_only_known']].all())
    plans={arm:load(folder/'basis_plan.npz') for arm,folder in folders.items()}
    checks['all_shared_basis_plan_buffers_identical']=plans['I0'].keys()==plans['W4'].keys() and all(byte_equal(v,plans['W4'][k]) for k,v in plans['I0'].items())
    checks['plans_keep_common_eligible_scope']=all(np.array_equal(v['eligible'],eligible) and np.array_equal(v['world_row_ids'],active) and str(v['source_role'])=='qualified_common_actual_alphaT' for v in plans.values())
    states={};initial={};training={};budget={};node_bounds={};parameterization_reconstruction={}
    for arm,folder in folders.items():
        states[arm]=checkpoint(folder/'checkpoint000200.pt');initial[arm]=checkpoint(folder/'checkpoint000000.pt')
        saved=states[arm];zero=initial[arm];training[arm]=read(folder/'training.json');completed=read(folder/'completed.json')
        trace,queue,sampler=expected_trace(200)
        checks[arm+'_200_actual_updates_complete']=completed['completed'] and completed['formal'] and completed['step']==200 and completed['actual_updates']==200 and len(training[arm])==200 and [x['step'] for x in training[arm]]==list(range(1,201))
        checks[arm+'_all_training_scalars_finite']=all(np.isfinite(value) for record in training[arm] for value in record.values() if isinstance(value,(int,float)))
        probe=read(folder/'gradient_probe.json')
        checks[arm+'_native_gradient_receipt_pass']=probe['actual_native_raster'] and probe['opacity_scale_geometry_parameters']==0 and all(row['finite'] and row['S_gradient_max']>0 and row['total_gradient_max']>0 for row in probe['rows'])
        checks[arm+'_trace_matches_seed101_queue']=saved['trace']==trace and saved['queue']==queue and saved['sampler']==sampler and [x['raw'] for x in training[arm]]==trace
        checks[arm+'_saved_checkpoint_step_and_contract']=saved['step']==200 and saved['formal_run'] and saved['contract']==contracts[arm]
        checks[arm+'_initial_delta_exact_zero']=zero['step']==0 and bool((zero['model']['delta_values']==0).all()) and not zero['optimizer']['state']
        opt=saved['optimizer'];checks[arm+'_Adam_step200_one_color_group']=len(opt['param_groups'])==1 and len(opt['param_groups'][0]['params'])==1 and all(int(x['step'])==200 for x in opt['state'].values())
        checks[arm+'_only_delta_model_mutated']=saved['model'].keys()==zero['model'].keys() and all(torch.equal(value,zero['model'][k]) for k,value in saved['model'].items() if k!='delta_values')
        correction=load(folder/'learned_correction.npz');delta=correction['delta_values'];p=plans[arm]
        checks[arm+'_saved_delta_matches_checkpoint']=byte_equal(delta,saved['model']['delta_values'].numpy())
        checks[arm+'_finite_node_bounds']=bool(np.isfinite(delta).all()) and bool((np.abs(delta)<=1).all())
        lower=delta<=-1;upper=delta>=1
        node_bounds[arm]=dict(shape=list(delta.shape),min=float(delta.min()),max=float(delta.max()),active_lower_scalars=int(lower.sum()),active_upper_scalars=int(upper.sum()),active_bound_scalars=int((lower|upper).sum()),active_bound_nodes=int((lower|upper).any(1).sum()),near_bound_1e_6_scalars=int((np.abs(delta)>=1-1e-6).sum()))
        if arm=='I0':row_delta=np.zeros_like(p['base_srgb']);row_delta[p['eligible']]=delta
        else:
            field=np.zeros((int(np.prod(p['shape'])),3),np.float32);field[p['mutable_texels']]=delta
            row_delta=(field[p['corner_ids']].astype(np.float64)*p['weights'][...,None]).sum(1).astype(np.float32)*p['eligible'][:,None]
        reconstructed=np.clip(p['base_srgb']+row_delta,0,1)
        maxerr=float(np.abs(reconstructed-correction['effective_rgb']).max())
        checks[arm+'_reconstructed_effective_shader_matches']=maxerr<=1e-6
        checks[arm+'_all_noneligible_colors_exact_c0']=byte_equal(correction['effective_rgb'][~eligible],p['base_srgb'][~eligible])
        parameterization_reconstruction[arm]=dict(max_abs=maxerr,RGB_clipped_0_1=bool(((correction['effective_rgb']>=0)&(correction['effective_rgb']<=1)).all()),shared_node_box_is_declared_method_constraint=True)
        budget[arm]=dict(step=saved['step'],training_records=len(training[arm]),frame_counts={str(raw):saved['trace'].count(raw) for raw in [86,110]},Adam_steps=[int(x['step']) for x in opt['state'].values()])
    checks['same_complete_sampling_trace']=states['I0']['trace']==states['W4']['trace']
    for name in ['torch_rng','cuda_rng','numpy_rng','python_rng','sampler','queue','trace']:
        checks['pair_final_'+name+'_exact']=not compare_snapshot(states['I0'][name],states['W4'][name],name)
    canonical=c.parent/'road_asset_20261116_r1/phase2/formal_r1/asset';asset_checks={};persistence={}
    for arm,folder in folders.items():
        changed=[];unchanged=[]
        with np.load(canonical/'road.npz',allow_pickle=False) as source,np.load(folder/'asset/road.npz',allow_pickle=False) as exported:
            checks[arm+'_road_field_set_same']=source.files==exported.files
            for key in source.files:
                a,b=source[key],exported[key]
                if byte_equal(a,b):unchanged.append(key)
                else:changed.append(key)
                if key!='reference_srgb':checks[arm+'_road_frozen_'+key]=byte_equal(a,b)
            baseRGB=source['reference_srgb'];rgb=exported['reference_srgb'];world_eligible=scope['world_eligible_color']
            checks[arm+'_world_RGB_outside_eligible_exact']=byte_equal(baseRGB[~world_eligible],rgb[~world_eligible])
            correction=load(folder/'learned_correction.npz')
            checks[arm+'_baked_RGB_matches_effective_shader']=byte_equal(rgb[active],correction['effective_rgb'])
            known_changed=int(np.any(rgb[active[scope['S_only_known']]]!=baseRGB[active[scope['S_only_known']]],axis=1).sum())
        prior_changed=[];prior_unchanged=[]
        with np.load(canonical/'provenance.npz',allow_pickle=False) as source,np.load(folder/'asset/provenance.npz',allow_pickle=False) as exported:
            for key in source.files:
                ok=key in exported.files and byte_equal(source[key],exported[key])
                (prior_unchanged if ok else prior_changed).append(key)
                if key not in ['effective_row_rgb','asset_mode']:checks[arm+'_immutable_prior_'+key]=ok
            checks[arm+'_capacity_source_prior_c0_exact']=byte_equal(exported['capacity_source_prior_reference_srgb'],baseRGB)
            checks[arm+'_capacity_effective_RGB_exact']=byte_equal(exported['capacity_effective_rgb'],rgb) and byte_equal(exported['effective_row_rgb'],rgb)
        asset_checks[arm]=dict(road_changed_fields=changed,road_unchanged_fields=unchanged,source_prior_changed_fields=prior_changed,source_prior_unchanged_count=len(prior_unchanged),S_only_known_effective_rows_changed=known_changed)
        K=read(folder/'joint_K.json');reload=read(folder/'public_reload.json');persist=read(folder/'shadow_known_persistence.json')
        checks[arm+'_native_K_receipt_exact_77266']=K['all_exact'] and K['protected_pixel_frames']==77266 and [x['raw'] for x in K['rows']]==list(FRAMES) and all(x['RGB_max']==0 and x['changed_pixels']==0 for x in K['rows'])
        checks[arm+'_public_reload_native_receipt_pass']=reload['public_loader'] and reload['GPU_executed'] and [x['raw'] for x in reload['rows']]==list(FRAMES) and all(max(x['RGB_max'],x['alpha_max'])<=1e-6 for x in reload['rows'])
        checks[arm+'_S_only_known_persistence_actual']=known_changed==persist['effective_RGB_changed_rows'] and persist['S_only_known_eligible']==3637 and all(persist['prior_ledger_fields_exact'].values())
        persistence[arm]=dict(joint_K_native_receipt=K,public_loader_native_receipt=reload,known_effective_override=persist,
            verification_boundary='existing native execution receipts and independent saved-row immutability are checked; CPU audit performs no fresh renderer replay')
    masks=load(pair/'frozen_frequency_masks.npz');frequency_meta=read(pair/'frozen_frequency_masks.json');teacher_rows=[];frequency_rows=[];coverage_rows=[]
    eval_difference_max=0.;fixedtotal=0;fulltotal=0
    for raw in FRAMES:
        geo=load(c/f'ideal_targets_r2/geometry/{raw:03d}.npz');rgb0=geo['baseline_forward_rgb_srgb__not_teacher'];alpha=geo['fixed_alpha'];full=load(c/f'ideal_targets_r2/constant/{raw:03d}.npz')['mask']
        md,mv,mh=[masks[f'{raw:03d}_{kind}'] for kind in ['domain','vertical','horizontal']]
        checks[f'raw{raw}_frozen_domain_from_canonical_alpha']=np.array_equal(md,full&(alpha>.9))
        checks[f'raw{raw}_stencils_three_valid_pixels']=np.array_equal(mv,md[2:]&md[1:-1]&md[:-2]) and np.array_equal(mh,md[:,2:]&md[:,1:-1]&md[:,:-2])
        fixedtotal+=int(md.sum());fulltotal+=int(full.sum());rgb_by_arm={'baseline':rgb0}
        for arm,folder in folders.items():
            output=load(folder/'rendered'/f'{raw:03d}.npz');rgb_by_arm[arm]=output['rgb_srgb']
            checks[f'{arm}_raw{raw}_coverage_fixed_bitwise']=byte_equal(output['alpha'],alpha)
            coverage_rows.append(dict(raw=raw,arm=arm,frozen_full_pixels=int(full.sum()),support=int((full&(output['alpha']>.9)).sum()),gap=int((full&(output['alpha']<.05)).sum()),thin=int((full&(output['alpha']>=.05)&(output['alpha']<=.9)).sum())))
        metrics={key:frequency_metrics(value,rgb0,md,mv,mh) for key,value in rgb_by_arm.items()}
        freq=dict(raw=raw,role='train' if raw in [86,110] else 'diagnostic',domain_pixels=int(md.sum()),vertical_stencils=int(mv.sum()),horizontal_stencils=int(mh.sum()),metrics=metrics,
            excess={key:excess_gate(metrics['baseline'][key],metrics['I0'][key],metrics['W4'][key]) for key in ['E_vertical','E_horizontal','RGB_delta_energy']},hard_gate_metric='E_vertical')
        frequency_rows.append(freq)
        for arm in folders:
            reported=read(folders[arm]/'evaluation_after.json');r=next(x for x in reported['rows'] if x['raw']==raw)
            for key,reported_key in [('E_vertical','vertical_E'),('E_horizontal','horizontal_E'),('RGB_delta_energy','RGB_delta_energy')]:eval_difference_max=max(eval_difference_max,abs(metrics[arm][key]-r[reported_key]))
        if raw in [86,110]:
            target=load(c/'supervision_v4_candidate_r1'/f'{raw:03d}.npz')
            for role,(mask,q,truth) in PHOTO_ROLES.items():
                errors={key:premult_mae(value,alpha,target[truth],target[mask],target[q]) for key,value in rgb_by_arm.items()}
                teacher_rows.append(dict(raw=raw,domain=role,photo_pixels=int(target[mask].sum()),fixed_q_sum=float((target[mask]*target[q]).sum()),errors=errors,
                    W4_minus_I0=errors['W4']-errors['I0'],noninferiority_margin=1/255,noninferiority_pass=errors['W4']<=errors['I0']+1/255,
                    I0_relative_improvement=(errors['baseline']-errors['I0'])/errors['baseline'],W4_relative_improvement=(errors['baseline']-errors['W4'])/errors['baseline']))
                for arm in folders:
                    r=next(x for x in read(folders[arm]/'evaluation_after.json')['rows'] if x['raw']==raw)
                    eval_difference_max=max(eval_difference_max,abs(errors[arm]-r['metrics'][role]))
    checks['frozen_full38416_frequency37404_preserved']=fixedtotal==37404 and fulltotal==38416
    checks['independent_metrics_agree_exported_records']=eval_difference_max<1e-6
    smoke=pair/'smoke_r1';engineering=[]
    for name in ['save2','restore3','replay3','finish4','W4_save1','W4_restore2']:
        folder=smoke/name;done=read(folder/'completed.json');records=read(folder/'training.json');snapshot=checkpoint(folder/f"checkpoint{done['step']:06d}.pt")
        checks[name+'_engineering_actual_count_matches_records']=not done['formal'] and not snapshot['formal_run'] and done['actual_updates']==len(records) and records[-1]['step']==done['step']
        if name in ['restore3','replay3','finish4','W4_restore2']:
            checks[name+'_full_restore_record']=read(folder/'restore.json')['full_state_restored_exact']
        engineering.append(dict(run=name,arm=done['arm'],actual_updates=done['actual_updates'],end_step=done['step'],trace_length=len(snapshot['trace']),formal=False))
    checks['engineering_updates_sum8']=sum(x['actual_updates'] for x in engineering)==8
    a=checkpoint(smoke/'restore3/checkpoint000003.pt');b=checkpoint(smoke/'replay3/checkpoint000003.pt');replay_diff=compare_snapshot(a,b)
    checks['lifecycle_next_update_replay_only_small_numeric_differences']=all(x['type']=='tensor' and (x['path']=='/model/delta_values' or x['path'].startswith('/optimizer/state/')) and x['max_abs']<=1e-6 for x in replay_diff)
    pair_done=read(pair/'pair_completed.json');checks['parent400plus8actual_budget']=pair_done['completed'] and pair_done['formal_updates']==400 and pair_done['engineering_updates']==8 and not pair_done['production_promoted']
    numeric_fit=all(x['noninferiority_pass'] for x in teacher_rows);applicable=[x for x in frequency_rows if x['excess']['E_vertical']['pass_gate'] is not None]
    stripe_pass=bool(applicable) and all(x['excess']['E_vertical']['pass_gate'] for x in applicable)
    engineering_pass=all(checks.values())
    result=dict(schema='road_capacity_real_pair_independent_CPU_acceptance.v1',completed=True,CPU_only=True,new_GPU_operations=0,new_render_calls=0,formal_outputs_modified=False,
        status='ENGINEERING_PASS_NUMERIC_JOINT_GATE_FAIL' if engineering_pass and not(numeric_fit and stripe_pass) else 'PASS' if engineering_pass else 'ENGINEERING_AUDIT_FAILED',
        summary=dict(engineering_acceptance=engineering_pass,engineering_checks=len(checks),teacher_noninferiority_pass=numeric_fit,teacher_noninferiority_failures=[dict(raw=x['raw'],domain=x['domain'],delta=x['W4_minus_I0']) for x in teacher_rows if not x['noninferiority_pass']],
            vertical_positive_excess_half_gate_pass=stripe_pass,vertical_applicable_frames=[x['raw'] for x in applicable],vertical_NA_frames=[x['raw'] for x in frequency_rows if x['excess']['E_vertical']['pass_gate'] is None],
            combined_numeric_gate_pass=numeric_fit and stripe_pass,formal_updates=400,engineering_updates=8,production_promoted=False),
        checks=checks,failed_engineering_checks=[k for k,v in checks.items() if not v],input_pins=pins,scope_rows=int(eligible.sum()),scope_S_only_known=int(scope['S_only_known'].sum()),
        teacher_metrics=teacher_rows,frequency_metrics=frequency_rows,coverage=coverage_rows,frozen_full_domain_pixels=fulltotal,frozen_frequency_domain_pixels=fixedtotal,
        node_bounds=node_bounds,shader_reconstruction=parameterization_reconstruction,asset_immutability=asset_checks,persistence=persistence,
        sampler_and_formal_budget=budget,engineering_lifecycle=engineering,next_update_replay_differences=replay_diff,independent_vs_runner_metric_max_abs=eval_difference_max,
        interpretation=dict(teacher='same frozen pseudo C/S and real visible A; no hidden road GT',frequency='primary hard gate is vertical excess; horizontal/delta ratios are reported diagnostics',
            W4_bound='counted active scalar/node bounds document declared bounded W4; no unbounded control isolates the cause',
            visual_verdict=None,human_verdict=None,I_reg_executed=False),
        native_evidence_boundary='K and public reload are audited existing real native receipts and saved-row invariants; no new independent GPU replay is performed',
        GPU_keep_on=True)
    write(args.output,result)
    print(json.dumps(dict(status=result['status'],summary=result['summary'],node_bounds=node_bounds,output=str(args.output)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
