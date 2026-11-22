"""Independent all-leaf exact restore / registered bounded replay audit; no torch."""
import argparse,ast,copy,json
from pathlib import Path
import numpy as np
from verify_joint_asset_cpu import pin,read,write,array_record,SOURCE_SHA


def audit_buffers(directory,strict):
    manifest_path=directory/'state_comparison.json';manifest=read(manifest_path)
    left_path=directory/'expected_state.npz';right_path=directory/'replay_state.npz'
    left=np.load(left_path,allow_pickle=False);right=np.load(right_path,allow_pickle=False)
    leaves=manifest['leaves'];structures=manifest['structures'];checks={}
    checks['NPZ_file_pins_exact']=pin(left_path)['sha256']==manifest['expected_npz_sha256'] and pin(right_path)['sha256']==manifest['replay_npz_sha256']
    checks['all_NPZ_keys_exact_full_leaf_manifest']=set(left.files)==set(right.files)=={x['key'] for x in leaves}
    checks['leaf_keys_and_paths_unique']=len(leaves)==len({x['key'] for x in leaves})==len({x['path'] for x in leaves})
    checks['strict_vs_bounded_mode_matches']=manifest['strict_restore'] is strict
    checks['epsilon_multiplier_still_four']=manifest['epsilon_multiplier']==4
    allpaths={x['path'] for x in leaves}|{x['path'] for x in structures};graph_ok=True
    for node in structures:
        prefix=node['path']+'/';children={p[len(prefix):].split('/')[0] for p in allpaths if p.startswith(prefix)}
        kind=node['kind']
        if kind=='dict':expected={str(ast.literal_eval(k)) for k in node['keys']}
        elif kind in ['tuple','list']:expected={str(i) for i in range(node['length'])}
        elif kind=='None':expected=set()
        else:graph_ok=False;continue
        if children!=expected:graph_ok=False
    checks['all_declared_structure_children_present']=graph_ok
    top=next(x for x in structures if x['path']=='')
    checks['complete_top_state_fields']=set(ast.literal_eval(k) for k in top['keys'])=={'parameters','optimizer','sampler','logical_step','python_rng','numpy_rng','torch_rng','cuda_rng','source_sha256','guard_sha256'}
    rows=[];parameter_max=0.;optimizer_max=0.;maxratio=0.;fullbytes=True
    for leaf in leaves:
        key=leaf['key'];path=leaf['path'];a=left[key];b=right[key];floating=a.dtype.kind in 'fc'
        counter=(path.startswith('/sampler') or 'rng' in path.lower() or path.endswith('/logical_step') or path.endswith('/step'))
        # Optimizer hyperparameters are configuration, not mutable moment buffers.
        hyper=path.startswith('/optimizer/param_groups/')
        exact=strict or counter or hyper or not floating
        shape_dtype=a.dtype==b.dtype and a.shape==b.shape and str(a.dtype)==leaf['dtype'] and list(a.shape)==leaf['shape'] and a.size==leaf['elements']
        samebytes=shape_dtype and a.tobytes()==b.tobytes();fullbytes&=samebytes
        finite=not floating or (np.isfinite(a).all() and np.isfinite(b).all())
        maximum=0.;ratio=0.
        if floating:
            kind=np.complex128 if a.dtype.kind=='c' else np.float64
            delta=abs(a.astype(kind)-b.astype(kind));bound=4*np.finfo(a.dtype).eps*(1+abs(a.astype(kind)))
            maximum=float(delta.max()) if a.size else 0.;ratio=float((delta/bound).max()) if a.size else 0.
            passed=shape_dtype and finite and (samebytes if exact else bool((delta<=bound).all()))
        else:passed=shape_dtype and samebytes
        pins=(array_record(a)['C_order_bytes_sha256']==leaf['expected_bytes_sha256'] and array_record(b)['C_order_bytes_sha256']==leaf['replay_bytes_sha256'])
        policy=(leaf['required_exact']==(strict or counter or not floating)) and leaf['counter_or_rng']==counter
        if path.startswith('/parameters/'):parameter_max=max(parameter_max,maximum)
        if path.startswith('/optimizer/state/'):optimizer_max=max(optimizer_max,maximum)
        maxratio=max(maxratio,ratio)
        rows.append({'path':path,'dtype':a.dtype.str,'shape':list(a.shape),'elements':int(a.size),
          'strict_or_semantic_exact':exact,'optimizer_hyperparameter_exact':hyper,'byteexact':samebytes,
          'all_finite':bool(finite),'max_abs':maximum,'max_error_over_registered_bound':ratio,
          'producer_leaf_pins_exact':pins,'producer_policy_matches_registered_rule':policy,
          'independent_pass':bool(passed and pins and policy)})
    checks['all_leaves_independently_PASS']=all(x['independent_pass'] for x in rows)
    source=next(x for x in leaves if x['path']=='/source_sha256');guard=next(x for x in leaves if x['path']=='/guard_sha256')
    checks['source_SHA_leaf_exact']=str(left[source['key']].item())==SOURCE_SHA
    return {'directory':str(directory),'manifest':pin(manifest_path),'expected':pin(left_path),'replay':pin(right_path),
      'checks':checks,'decision':'PASS' if all(checks.values()) else 'FAIL','leaf_count':len(rows),
      'structure_node_count':len(structures),'total_elements':sum(x['elements'] for x in rows),
      'all_state_leaf_bytes_exact':bool(fullbytes),'parameter_max_abs':parameter_max,'optimizer_state_max_abs':optimizer_max,
      'maximum_error_over_registered_bound':maxratio,'guard_SHA_leaf':str(left[guard['key']].item()),'leaves':rows}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--v1-run',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=read(a.v1_run/'fit_contract.json');revised=read(a.contract);normalized=copy.deepcopy(revised)
    normalized['schema']=original['schema'];normalized.pop('replay_validation')
    protocol=revised['replay_validation'];summary=read(a.run/'restore_replay.json');old=read(a.v1_run/'restore_replay.json')
    restore=audit_buffers(a.run/'checkpoint_restore',True);replay=audit_buffers(a.run/'cuda_replay',False)
    guard=read(a.root/'joint_persistent_asset_audit_v1/frozen_invariants.json')['guard']['sha256']
    checks={'formal_and_all_method_configuration_exact':normalized==original,
      'formal_steps_still400':revised['formal_steps']==400,'train86_110_eval134_179_198_exact':revised['train_frames']==[86,110] and revised['evaluation_only_frames']==[134,179,198],
      'protocol_eps_four':protocol['eps_multiplier']==4,'protocol_loss_bound1e_7':protocol['loss_abs_bound']==1e-7,
      'strict_restore_all_leaves_PASS':restore['decision']=='PASS' and restore['all_state_leaf_bytes_exact'],
      'bounded_replay_all_leaves_PASS':replay['decision']=='PASS',
      'source_guard_same':restore['guard_SHA_leaf']==replay['guard_SHA_leaf']==guard,
      'same_replay_training_view':summary['replay_same_raw'] is True,
      'loss_within_original_registered_bound':summary['replay_loss_max_abs']<=protocol['loss_abs_bound'],
      'v1_strict_FAIL_retained':old['replay_parameters_optimizer_RNG_bitexact'] is False,
      'current_summary_bitexact_truthful':summary['replay_parameters_optimizer_RNG_bitexact']==replay['all_state_leaf_bytes_exact'],
      'current_summary_parameter_max_matches':summary['parameter_max_abs']==replay['parameter_max_abs']}
    result={'schema':'R9.independent_joint_smoke_state_replay_audit.v2','complete':True,
      'decision':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'original_contract':pin(a.v1_run/'fit_contract.json'),
      'revised_lifecycle_contract':pin(a.contract),'engineering_protocol_revision_not_formal_method_change':True,
      'original_v1_strict_replay_failure':pin(a.v1_run/'restore_replay.json'),'original_v1_resources':pin(a.v1_run/'resources.json'),
      'strict_restore':restore,'bounded_next_step_replay':replay,'loss_max_abs':summary['replay_loss_max_abs'],
      'float_rule':'Every permitted float element <=4eps(dtype)*(1+abs(reference)); finite and same dtype/shape; exact RNG/counter/step/nonfloat/optimizer hyperparameters.',
      'CPU_only':True,'torch_imported':False,'GPU_forward':0,'diffusion_calls':0,'updates':0}
    write(a.output,result);print(json.dumps({'decision':result['decision'],'checks':checks,'restore_leaves':restore['leaf_count'],
      'replay_leaves':replay['leaf_count'],'parameter_max_abs':replay['parameter_max_abs'],'optimizer_max_abs':replay['optimizer_state_max_abs'],
      'maximum_error_over_bound':replay['maximum_error_over_registered_bound'],'output':pin(a.output)}))
    if result['decision']!='PASS':raise SystemExit(2)


if __name__=='__main__':main()
