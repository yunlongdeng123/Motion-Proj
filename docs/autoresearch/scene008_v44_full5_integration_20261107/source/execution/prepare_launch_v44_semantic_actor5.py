"""Seal and launch the explicitly released four-frame semantic-sidecar composition."""
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,time
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_full5_integration_20261107_r1')
ADAPTER=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42/scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py')
NEW_SHA='08bcf4ac5e58f59bb1a2f917bdfd49fd0609c8b56e1d344528811738fd2ab6ee'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ADAPTER)==NEW_SHA
parent=ROOT/'contract/execution_contract_actor5.json';assert sha(parent)=='2e5b50c7856bd053c1fd95cf4d0f2466542bdb6b464e57a1f477ce059254eb00'
manifest=ROOT/'semantic_actor5_calibration4/semantic_actor5_manifest.json';sidecar=json.loads(manifest.read_text())
selection=manifest.parent/sidecar['selection_file'];assert sha(selection)==sidecar['selection_sha256']
assert sidecar['raw_ids']==[94,105,130,137] and sidecar['threshold']=={'purity_strictly_greater_than':.5,'total_mass_strictly_greater_than':0.}
calib=json.loads((manifest.parent/'receipt.json').read_text());assert calib['joint_forwards']==4 and calib['dummy_feature_backwards']==4
assert calib['selected_rows']=={'static':160,'dynamic':56118} and calib['model_parameters_environment_and_original_identity_unchanged']
contract=copy.deepcopy(json.loads(parent.read_text()));contract['created_unix']=time.time();contract['case_id']='SEMANTIC_ACTOR5_FIXED_VISIBLE_MASS_SIDECAR_PILOT'
contract['parent_contract_sha256']=sha(parent);contract['pilot_raws']=[94,105,130,137]
contract['prior_case']='Source-owner add49 and strict legacy-static retirement4 failed quality; both preserved'
candidate=contract['candidate'];candidate['adapter_sha256']=NEW_SHA;candidate['api']='SemanticActor5V44Integration'
candidate['semantic_sidecar']={'manifest':str(manifest),'manifest_sha256':sha(manifest),'selection':str(selection),'selection_sha256':sha(selection),
 'schema':sidecar['schema'],'threshold':sidecar['threshold'],'selected_rows':calib['selected_rows'],
 'source_dynamic_owner_rows':{'5':54686,'37':1432,'6':0,'18':0,'26':0},'static_rows':160,
 'ancestry_and_original_owner_ids_preserved':True,'source37_selected_rows_are_query_semantic5':True,
 'QUERY_reads_BUILD_masks':False,'calibration_role':'Derived from original BUILD actor5 masks on four fixed views; not held-out test validation'}
candidate['identities_from_checkpoint_only']=False
candidate['semantic_query_identity_from_fixed_visibility_sidecar']=True
candidate['retire_legacy_static_actor5']=False
candidate['v44_all_original_rows_masks_colours_gates_environment_and_C2_unchanged']=True
contract['domains']['pilot_a5_pixels']={'94':2052,'105':10773,'130':0,'137':0}
contract['domains']['pilot_upper_pixels']={'94':471,'105':2091,'130':0,'137':0}
contract['numeric_gates']['raw94_a5_mask_MAE_delta_strictly_below']=0.
contract['numeric_gates']['raw94_a5_upper_MAE_delta_strictly_below']=0.
contract['numeric_gates']['raw105_a5_upper_MAE_delta_strictly_below']=0.
contract['numeric_gates']['additional_strict_a5_gates_states']=['keep','delete']
contract['exact_guards']['semantic_rows_exact_match_fixed_sidecar_ids']=True
contract['exact_guards']['no_source_owner37_zero_assertion_for_semantic5_rows']=True
contract['visual_gates']['raw94_and105_a5_full_and_upper_review_required']=True
destination=ROOT/'contract/execution_contract_semantic_actor5.json';assert not destination.exists();destination.write_text(json.dumps(contract,indent=2)+'\n')
source=(ROOT/'source/render_v44_actor5_candidate4.py').read_text()
changes={
 '59a94458a3423d1adb4e397f8bc1081029f2250d4b7a3b8087427cff8bb3934a':NEW_SHA,
 'contract/execution_contract_actor5.json':'contract/execution_contract_semantic_actor5.json',
 'from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import Full5V44Integration':'from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration',
 "rt=Full5V44Integration(baseline=base,full5_transform_path=ROOT/'contract/full5_transform_matrix.txt',added_actor_ids=(5,))":
 "rt=SemanticActor5V44Integration(base,semantic_sidecar_path=ROOT/'semantic_actor5_calibration4/semantic_actor5_manifest.json',full5_transform_path=ROOT/'contract/full5_transform_matrix.txt')\n"
 " with np.load(ROOT/'semantic_actor5_calibration4/semantic_actor5_rows.npz',allow_pickle=False) as z:\n"
 "  assert np.array_equal(rt.semantic_static_row_ids.detach().cpu().numpy(),z['static_row_ids'])\n"
 "  assert np.array_equal(rt.semantic_dynamic_row_ids.detach().cpu().numpy(),z['dynamic_row_ids'])\n"
 " assert len(rt.semantic_static_row_ids)==160 and len(rt.semantic_dynamic_row_ids)==56118",
 "outdir=ROOT/'candidate_a5_4'":"outdir=ROOT/'candidate_semantic_a5_4'",
 "b=before['rows'];c=scene['rows'];n=len(b['means3D']);assert len(c['means3D'])>=n":"b=before['rows'];c=scene['rows'];n=len(b['means3D']);assert len(c['means3D'])==n+56278",
 'RENDERED_NATIVE_SHARED_SORT_CANDIDATE4_PENDING_PAIRED_GATES_AND_ROOT_REVIEW':'RENDERED_FIXED_SEMANTIC_ACTOR5_SIDECAR_PILOT_PENDING_GATES_AND_ROOT_REVIEW',
}
for before,after in changes.items():
 assert source.count(before)==1,(before,source.count(before));source=source.replace(before,after)
assert source.count('[105,113,130,137]')==3;source=source.replace('[105,113,130,137]','[94,105,130,137]')
ast.parse(source);worker=ROOT/'source/render_v44_semantic_actor5_candidate4.py';assert not worker.exists();worker.write_text(source)
assert sha(ADAPTER)==NEW_SHA and sha(manifest)==contract['candidate']['semantic_sidecar']['manifest_sha256']
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
command=[sys.executable,'-B',str(worker)]
with (ROOT/'candidate_semantic_a5_4.log').open('x') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env,start_new_session=True)
receipt={'status':'STARTED_FIXED_SEMANTIC_ACTOR5_SIDECAR_PILOT','pid':p.pid,'unix':time.time(),'command':command,
 'adapter_sha256':NEW_SHA,'contract_path':str(destination),'contract_sha256':sha(destination),'worker_sha256':sha(worker),
 'semantic_manifest_sha256':sha(manifest),'semantic_selection_sha256':sha(selection),'raws':[94,105,130,137],
 'new_training_updates':0,'all49_auto_release':False,'shutdown':False,'reset_card':False}
(ROOT/'contract/candidate_semantic_actor5_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
