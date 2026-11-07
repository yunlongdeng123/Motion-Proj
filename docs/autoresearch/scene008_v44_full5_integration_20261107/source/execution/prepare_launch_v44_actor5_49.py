from pathlib import Path
import json,hashlib,time,copy,subprocess,sys,os,ast
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
ADAPTER=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py')
SHA='59a94458a3423d1adb4e397f8bc1081029f2250d4b7a3b8087427cff8bb3934a'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ADAPTER)==SHA and not (ROOT/'candidate_a5_49').exists()
qa_path=ROOT/'review/cpu_independent_candidate_a5_4.json';qa=json.loads(qa_path.read_text())
assert qa['status']=='PASS_SAVED_FLOAT_ENGINEERING_AND_NUMERIC_GATES' and len(qa['numeric_gates'])==19 and not qa['failed_gates']
sh=ROOT/'review/sh_precompute_raw130_retry1/receipt.json';assert sh.is_file()
parent=ROOT/'contract/execution_contract_actor5.json';original_sha=sha(parent)
assert original_sha=='2e5b50c7856bd053c1fd95cf4d0f2466542bdb6b464e57a1f477ce059254eb00'
contract=copy.deepcopy(json.loads(parent.read_text()));contract['created_unix']=time.time()
contract['case_id']='ACTOR5_ONLY_ROOT_RELEASED_ALL49';contract['parent_pilot_contract_sha256']=original_sha
contract['evaluation_scope']='candidate49_keep_delete';contract['evaluation_raws']=list(range(89,138))
contract['baseline']['reuse_formal_all49_without_new_forward']=True
contract['all49_release']=dict(automatic=False,root_explicitly_approved=True,
    required_images_reviewed=[105,113,130,137],numeric_gate_file=str(qa_path),numeric_gate_sha256=sha(qa_path),
    shader_control_receipt=str(sh),shader_control_sha256=sha(sh),shader_rgb_maxabs=1.1920928955078125e-7,shader_alpha_maxabs=0.,
    authority='Root explicitly released actor5-only49 after full4 KEEP/DELETE boards,105crop,19numeric gates and real SHcontrol')
path=ROOT/'contract/execution_contract_actor5_49.json';assert not path.exists();path.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
source=(ROOT/'source/render_v44_actor5_candidate4.py').read_text()
changes={
 "contract/execution_contract_actor5.json":"contract/execution_contract_actor5_49.json",
 "for raw in [105,113,130,137]:":"for raw in range(89,138):",
 "outdir=ROOT/'candidate_a5_4'":"outdir=ROOT/'candidate_a5_49'",
 "raws=[105,113,130,137]":"raws=list(range(89,138))",
 "RENDERED_NATIVE_SHARED_SORT_CANDIDATE4_PENDING_PAIRED_GATES_AND_ROOT_REVIEW":"RENDERED_NATIVE_SHARED_SORT_ACTOR5_ALL49_PENDING_COMPLETE_GATES",
}
for before,after in changes.items():
 assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
needle=" assert sha(ADAPTER)==EXPECTED and (ROOT/'contract/root_baseline4_exact_png_audit.json').exists()"
assert needle in source
source=source.replace(needle,needle+"\n assert contract['all49_release']['root_explicitly_approved'] and contract['evaluation_raws']==list(range(89,138))")
ast.parse(source);worker=ROOT/'source/render_v44_actor5_49.py';assert not worker.exists();worker.write_text(source)
assert sha(parent)==original_sha
command=[sys.executable,'-B',str(worker)];env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (ROOT/'candidate_a5_49.log').open('x') as log:
 p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt=dict(status='STARTED_ROOT_RELEASED_ACTOR5_ONLY_ALL49',pid=p.pid,unix=time.time(),command=command,
    adapter_sha256=SHA,contract_path=str(path),contract_sha256=sha(path),parent_pilot_contract_unchanged=original_sha,
    raws=list(range(89,138)),states=['keep','delete'],new_candidate_forward_budget=98,baseline_new_forwards=0,
    new_training_updates=0,shutdown=False,reset_card=False)
(ROOT/'contract/candidate_actor5_49_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
