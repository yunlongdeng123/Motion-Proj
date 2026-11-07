from pathlib import Path
import json,hashlib,time,copy,subprocess,sys,os,ast
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
ADAPTER=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py')
NEW_SHA='59a94458a3423d1adb4e397f8bc1081029f2250d4b7a3b8087427cff8bb3934a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ADAPTER)==NEW_SHA
original=ROOT/'contract/execution_contract.json';parent_sha=sha(original)
assert parent_sha=='83bec3f33a2d7da7633089d03ee18bcd45b8f19bd6d4cfd258ee0b35cd7e1cb2'
contract=copy.deepcopy(json.loads(original.read_text()));contract['created_unix']=time.time()
contract['case_id']='ACTOR5_ONLY_SAME_PREREGISTERED_THRESHOLDS'
contract['parent_contract_sha256']=parent_sha;contract['prior_case']='candidate4 three retained actors; visual/protected FAIL, retained unchanged'
contract['candidate']['retained_actor_ids']=[5];contract['exact_guards']['added_owner_ids']=[5]
contract['candidate']['adapter_sha256']=NEW_SHA
contract['candidate']['actors18_26']='Keep original V44 representation; no new Full5 rows for18/26'
contract['exact_guards']['unused_added_groups18_26_exact_zero']=True
destination=ROOT/'contract/execution_contract_actor5.json';assert not destination.exists()
destination.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
source=(ROOT/'source/render_v44_candidate4.py').read_text()
changes={
 "88998b82ad472797b88f4a3fcacf22b9f87af078f5c20eb75774f8eb50b6666c":NEW_SHA,
 "contract/execution_contract.json":"contract/execution_contract_actor5.json",
 "full5_transform_path=ROOT/'contract/full5_transform_matrix.txt')":"full5_transform_path=ROOT/'contract/full5_transform_matrix.txt',added_actor_ids=(5,))",
 "outdir=ROOT/'candidate4'":"outdir=ROOT/'candidate_a5_4'",
 "    assert all(np.isfinite(x).all() for x in [rgb,alpha,depth_sum,groups,added_rgb])":"    assert all(np.isfinite(x).all() for x in [rgb,alpha,depth_sum,groups,added_rgb])\n    assert float(np.abs(groups[5:]).max())==0. and float(np.abs(added_rgb[1:]).max())==0.",
}
for before,after in changes.items():
 assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
ast.parse(source);worker=ROOT/'source/render_v44_actor5_candidate4.py';assert not worker.exists();worker.write_text(source)
assert sha(original)==parent_sha and not (ROOT/'candidate_a5_4').exists()
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
command=[sys.executable,'-B',str(worker)]
with (ROOT/'candidate_a5_4.log').open('x') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='STARTED_ACTOR5_ONLY_CANDIDATE4',pid=p.pid,unix=time.time(),command=command,
    adapter_sha256=NEW_SHA,contract_path=str(destination),contract_sha256=sha(destination),parent_contract_unchanged=parent_sha,
    added_actor_ids=[5],raws=[105,113,130,137],same_thresholds_as_three_actor_case=True,
    all49_auto_release=False,new_training_updates=0,shutdown=False,reset_card=False)
(ROOT/'contract/candidate_actor5_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
