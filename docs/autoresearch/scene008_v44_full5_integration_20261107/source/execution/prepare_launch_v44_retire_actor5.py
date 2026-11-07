"""Seal the source-lineage retirement pilot, then launch its sole GPU worker."""
from pathlib import Path
import argparse, ast, copy, hashlib, json, os, subprocess, sys, time
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
ADAPTER=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py')
args=argparse.ArgumentParser();args.add_argument('--adapter-sha',required=True);args.add_argument('--audit-sha',required=True);o=args.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ADAPTER)==o.adapter_sha
asset=ROOT/'review/legacy_static_actor5/strict_actor5_static_lineage.npz'
audit=ROOT/'review/legacy_static_actor5/audit.json'
assert sha(asset)=='57b1d6bc342e784a043d12b59e8d0e5fafccc13dcb5fb31f33388e65cf569e91'
assert sha(audit)==o.audit_sha
with np.load(asset,allow_pickle=False) as a:
 ids=a['current_static_retirement_rows'];assert ids.shape==(45577,) and len(np.unique(ids))==45577
 assert a['initial_static_lineage_ids'].shape==(1496,)
parent=ROOT/'contract/execution_contract_actor5.json'
assert sha(parent)=='2e5b50c7856bd053c1fd95cf4d0f2466542bdb6b464e57a1f477ce059254eb00'
contract=copy.deepcopy(json.loads(parent.read_text()));contract['created_unix']=time.time()
contract['case_id']='ACTOR5_ONLY_STRICT_SOURCE_STATIC_RETIREMENT_PILOT'
contract['parent_contract_sha256']=sha(parent);contract['pilot_raws']=[94,105,130,137]
contract['prior_case']='actor5-only add without retirement: all49 RGB pooled regression; preserved unchanged'
contract['candidate']['adapter_sha256']=o.adapter_sha
contract['candidate']['retire_legacy_static_actor5']=True
contract['candidate']['strict_static_retirement']={'asset':str(asset),'asset_sha256':sha(asset),'audit':str(audit),'audit_sha256':sha(audit),
 'initial_sampled_rows':1496,'initial_unique_rows':1442,'current_static_rows':1042776,'strict_current_descendants':45577,
 'road_gate_intersection':3439,'original_road_gates_applied_before_retirement':True,
 'operator':'temporary rows.mask[:1042776] = original_mask[:1042776] & ~strict_descendant_mask; no parameter or opacity edit',
 'same_retirement_in_keep_and_delete':True,'selection_uses_saved_source_lineage_only':True}
contract['domains']['pilot_a5_pixels']={'94':2052,'105':10773,'130':0,'137':0}
contract['domains']['pilot_upper_pixels']={'94':471,'105':2091,'130':0,'137':0}
contract['numeric_gates']['raw94_a5_mask_MAE_delta_strictly_below']=0.
contract['numeric_gates']['raw94_a5_upper_MAE_delta_strictly_below']=0.
contract['numeric_gates']['raw105_a5_upper_MAE_delta_strictly_below']=0.
contract['numeric_gates']['additional_strict_a5_gates_states']=['keep','delete']
contract['exact_guards'].pop('shared_physical_rows_and_colours_byte_equal')
contract['exact_guards']['shared_physical_rows_and_colours_except_registered_mask_byte_equal']=True
contract['exact_guards']['mask_only_difference_exact_XOR_original_mask_AND_strict_static_descendants']=True
contract['exact_guards']['strict_static_mask_source_asset_sha256']=sha(asset)
contract['visual_gates']['raw94_and105_a5_full_and_upper_review_required']=True
destination=ROOT/'contract/execution_contract_actor5_retire.json';assert not destination.exists()
destination.write_text(json.dumps(contract,indent=2)+'\n')
source=(ROOT/'source/render_v44_actor5_candidate4.py').read_text()
changes={
 '59a94458a3423d1adb4e397f8bc1081029f2250d4b7a3b8087427cff8bb3934a':o.adapter_sha,
 'contract/execution_contract_actor5.json':'contract/execution_contract_actor5_retire.json',
 'added_actor_ids=(5,))':"added_actor_ids=(5,),retire_legacy_static_actor5=True,legacy_static_actor5_audit_path=ROOT/'review/legacy_static_actor5/audit.json')",
 "outdir=ROOT/'candidate_a5_4'":"outdir=ROOT/'candidate_a5_retire_4'",
 "     if key=='features':continue":"     if key in ['features','mask']:continue",
 "    before_hash=tensor_hash(shared_before);after_hash=tensor_hash(shared_after);assert before_hash==after_hash":
 "    no_mask_before_hash=tensor_hash(shared_before);no_mask_after_hash=tensor_hash(shared_after);assert no_mask_before_hash==no_mask_after_hash\n"
 "    n_static=1042776;assert rt.legacy_static_row_offset==0 and n>=n_static\n"
 "    with np.load(ROOT/'review/legacy_static_actor5/strict_actor5_static_lineage.npz',allow_pickle=False) as saved: retirement_ids=saved['current_static_retirement_rows'].copy()\n"
 "    independent=torch.zeros(n_static,dtype=torch.bool,device=b['mask'].device);independent[torch.as_tensor(retirement_ids,device=independent.device)]=True\n"
 "    assert torch.equal(independent,rt.legacy_static_actor5_retirement_mask)\n"
 "    assert torch.equal(torch.as_tensor(retirement_ids,device=independent.device),rt.legacy_static_actor5_retirement_rows)\n"
 "    expected_mask=b['mask'].clone();expected_mask[:n_static]=expected_mask[:n_static]&~independent\n"
 "    assert torch.equal(c['mask'][:n],expected_mask)\n"
 "    actual_xor=b['mask']^c['mask'][:n];expected_xor=torch.zeros_like(b['mask']);expected_xor[:n_static]=b['mask'][:n_static]&independent\n"
 "    assert torch.equal(actual_xor,expected_xor)\n"
 "    retirement_mask_xor_count=int(actual_xor.sum().item());retirement_mask_expected_hash=tensor_hash({'mask':expected_mask});retirement_mask_actual_hash=tensor_hash({'mask':c['mask'][:n]});assert retirement_mask_expected_hash==retirement_mask_actual_hash\n"
 "    before_hash=tensor_hash({k:v for k,v in b.items() if k!='features' and v is not None});after_hash=tensor_hash({k:v[:n] for k,v in c.items() if k!='features' and v is not None})",
 'shared_physical_before_hash=before_hash,shared_physical_after_hash=after_hash,':
 'shared_physical_before_hash=before_hash,shared_physical_after_hash=after_hash,\n'
 '        shared_physical_except_mask_before_hash=no_mask_before_hash,shared_physical_except_mask_after_hash=no_mask_after_hash,\n'
 '        retirement_mask_xor_count=retirement_mask_xor_count,retirement_mask_expected_hash=retirement_mask_expected_hash,retirement_mask_actual_hash=retirement_mask_actual_hash,\n'
 '        retirement_asset_sha256=sha(ROOT/\'review/legacy_static_actor5/strict_actor5_static_lineage.npz\'),',
 'RENDERED_NATIVE_SHARED_SORT_CANDIDATE4_PENDING_PAIRED_GATES_AND_ROOT_REVIEW':'RENDERED_STRICT_SOURCE_STATIC_RETIREMENT_PILOT_PENDING_ALL_GATES_AND_ROOT_REVIEW',
}
for before,after in changes.items():
 assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
assert source.count('[105,113,130,137]')==3
source=source.replace('[105,113,130,137]','[94,105,130,137]')
ast.parse(source);worker=ROOT/'source/render_v44_actor5_retire_candidate4.py';assert not worker.exists();worker.write_text(source)
assert sha(parent)=='2e5b50c7856bd053c1fd95cf4d0f2466542bdb6b464e57a1f477ce059254eb00'
assert not (ROOT/'candidate_a5_retire_4').exists()
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
command=[sys.executable,'-B',str(worker)]
with (ROOT/'candidate_a5_retire_4.log').open('x') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt={'status':'STARTED_STRICT_SOURCE_STATIC_ACTOR5_RETIREMENT_PILOT','pid':p.pid,'command':command,'unix':time.time(),
 'adapter_sha256':o.adapter_sha,'contract_path':str(destination),'contract_sha256':sha(destination),'worker_sha256':sha(worker),
 'added_actor_ids':[5],'raws':[94,105,130,137],'all49_auto_release':False,'new_training_updates':0,'shutdown':False,'reset_card':False}
(ROOT/'contract/candidate_actor5_retire_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
