"""Finite3x3000 steps. Release the shared GPU slot after each arm. No later stages."""
from pathlib import Path
import argparse,os,subprocess,json,datetime,shutil,time
p=argparse.ArgumentParser();p.add_argument('--worker',choices=['A_box','B_semantic_motion','C_correctable']);a=p.parse_args()
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=r/'ownership_controls_r12_sam3';py='/root/autodl-tmp/envs/drivingeditor/bin/python';state=root/'pipeline_state.json'
def write(status,**kw):
 temp=state.with_suffix('.tmp');temp.write_text(json.dumps(dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));temp.replace(state)
def command(script,*args):subprocess.run([py,str(r/script),*map(str,args)],cwd=r,check=True)
try:
 if not a.worker:
  for arm in ['A_box','B_semantic_motion','C_correctable']:
   if (root/arm/'complete.json').exists():raise FileExistsError('do not silently replay completed arm')
   if arm=='C_correctable':
    while not (root/'C_correction_code_ready.json').exists():
     write('WAITING_C_CORRECTION_CODE_NO_GPU_HELD',arm=arm,reason='finish and verify staged actual-owner/split implementation; not an experimental failure');time.sleep(30)
   # A non-cooperating external task also gets priority before flock acquisition.
   while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
    write('WAITING_OTHER_GPU_TASK',next_arm=arm);time.sleep(30)
   write('WAITING_GPU_SLOT',arm=arm)
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0'
   subprocess.run(['flock','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock',py,str(r/'three_control_controller_r12.py'),'--worker',arm],cwd=r,env=env,check=True)
  write('WAITING_ROOT_FINITE_THREE_CONTROL_GATE',arms=['A_box','B_semantic_motion','C_correctable'],effective_updates_each=3000,actual_updates_total=9000,later_stages_started=False)
 else:
  arm=a.worker;run=root/arm;data=root/('data_A_box' if arm=='A_box' else 'data_B_semantic_motion');implementation=r/'implementation_r12/shadow_rebuild_v3/run.py'
  if run.exists():raise FileExistsError('no duplicate arm')
  write('RUNNING',arm=arm,phase='geometry_first1500',data=str(data))
  command('implementation_r12/shadow_rebuild_v3/run.py','train','--data',data,'--config',root/'common_config.json','--output',run,'--execute','--allow-partial-masks','--max-updates',1500)
  shutil.copyfile(run/'latest.pt',run/'gate_001500.pt')
  if arm=='C_correctable':
   write('RUNNING',arm=arm,phase='probability_and_mixed_boundary_review_zero_SGD')
   command('fullscene_ownership_audit_common_r9.py','--run','ownership_controls_r12_sam3/'+arm,'--step',1500,'--implementation','implementation_r12','--data','ownership_controls_r12_sam3/data_B_semantic_motion','--certified-road','--all-identities','--output','ownership_controls_r12_sam3/C_identity_001500_before')
   command('C_split_r12.py')
   command('fullscene_ownership_audit_common_r9.py','--run','ownership_controls_r12_sam3/C_split_001500','--step',1500,'--implementation','implementation_r12','--data','ownership_controls_r12_sam3/data_B_semantic_motion','--certified-road','--all-identities','--output','ownership_controls_r12_sam3/C_identity_001500_after_split')
   command('C_finalize_owner_r12.py')
  write('RUNNING',arm=arm,phase='geometry_second1500')
  command('implementation_r12/shadow_rebuild_v3/run.py','train','--data',data,'--config',root/'common_config.json','--output',run,'--resume',run/'latest.pt','--execute','--allow-partial-masks','--max-updates',1500)
  write('RUNNING',arm=arm,phase='full49_independent_assets_and_edits')
  command('r12_geometry_review.py','--arm',arm)
  (run/'complete.json').write_text(json.dumps(dict(global_step=3000,arm=arm,full49=True,physics_shadow_trained=False,quality_accepted=False)))
except BaseException as exc:
 write('FAILED_ENGINEERING',arm=a.worker,error=repr(exc),no_method_verdict=True);raise
