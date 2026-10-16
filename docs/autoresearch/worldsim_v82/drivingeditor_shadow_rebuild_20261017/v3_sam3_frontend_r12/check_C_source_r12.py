"""Actual BUILD schema + core invariant checks. No scene/model file is changed."""
from r12_C_common import *
import subprocess,os,ast,time
start=time.monotonic();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONPATH']=str(R/'implementation_r12/shadow_rebuild_v3')
test=subprocess.run(['/root/autodl-tmp/envs/drivingeditor/bin/python','-m','unittest','discover','-s',str(R/'implementation_r12/shadow_rebuild_v3/tests'),'-p','test_identity.py','-v'],env=env,capture_output=True,text=True);print(test.stdout);print(test.stderr);test.check_returncode()
for name in ['r12_C_common.py','C_split_r12.py','C_finalize_owner_r12.py']:ast.parse((R/name).read_text())
d=Dataset(ROOT/'data_B_semantic_motion','cpu',True);tr=np.load(ROOT/'data_B_semantic_motion/trajectory.npz');pool=dict(np.load(ROOT/'uncertain_candidate_pool.npz'));build_only(pool,d);records=[]
for a in TARGETS:
 b=body_bank(a);road=road_bank(a);build_only(b,d);build_only(road,d)
 for values in [b,road]:
  for fi in np.unique(values['source_frame']):
   ids=np.flatnonzero(values['source_frame']==fi)[::max(1,int((values['source_frame']==fi).sum())//20)]
   if len(ids) and not np.allclose(transform(values['xyz'][ids],tr['poses'][int(fi),a]),values['source_world'][ids],atol=1e-4):raise ValueError('source canonical/world conversion failed')
 records.append(dict(actor=a,actual_body_measurements=len(b['xyz']),actual_positive_world_road_candidates=len(road['xyz'])))
motion=motion_source_keys();assert not torch.cuda.is_initialized()
receipt=dict(status='C_SOURCE_AND_CORE_READY_FOR_FINITE1500_INTERVENTION',implementation='implementation_r12',scripts=['C_split_r12.py','C_finalize_owner_r12.py','r12_C_common.py'],source_schema_BUILD_checks=True,actual_local_world_coordinate_samples_checked=True,CPU_identity_tests=4,CPU_total_previous_tests=33,actual_1500_intervention_executed=False,actual_asset_quality_accepted=False,actor_data=records,unknown_pool=len(pool['xyz']),source_motion_keys={str(k):len(v) for k,v in motion.items()},CUDA_initialized=False,scene_training_updates=0,QUERY_selection=False,seconds=time.monotonic()-start,reviewer='root agent; no subagents',human_verdict=None)
write(ROOT/'C_correction_code_ready.json',receipt);print(json.dumps(receipt),flush=True)
