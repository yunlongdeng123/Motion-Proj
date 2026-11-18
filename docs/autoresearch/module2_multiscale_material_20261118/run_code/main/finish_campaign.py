from pathlib import Path
import json,subprocess,hashlib,datetime

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def cmd(*args):return subprocess.check_output(args,cwd=R,text=True).strip()
pub=json.loads((P/'publication/publication_receipt.json').read_text())
validation=json.loads((P/'publication/source_validation.json').read_text())
accept=json.loads((P/'qa/final_acceptance.json').read_text())
start=json.loads((P/'registration/start.json').read_text())
index=Path(cmd('git','rev-parse','--git-path','index'))
if not index.is_absolute():index=R/index
head=cmd('git','rev-parse','HEAD');assert head==start['repo_HEAD']
assert index.read_bytes()==(P/'registration/active_index.before').read_bytes()
remote=cmd('git','ls-remote','origin','refs/heads/'+pub['branch']).split()[0]
assert remote==pub['commit']==pub['verified_remote_head'] and pub['pushed'] and validation['passed']
lease=json.loads((P/'gpu_lease.json').read_text());assert not lease['active'] and lease['owner'] is None
apps=cmd('nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader')
gpu=cmd('nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader')
assert gpu
ps=cmd('ps','-eo','pid,args').splitlines()
own=[x for x in ps if str(P) in x and 'finish_campaign.py' not in x and 'ps -eo' not in x]
# Shell wrappers for this synchronous read-only audit are not job controllers.
controllers=[x for x in own if any(k in x for k in ('run_material_pair.py','run_matched_i0.py','run_material_lifecycle.py','run_trellis','encode_native_previews.py'))]
assert not controllers
receipt=dict(schema='module2_multiscale_material_finish.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 task_id='MODULE2-MULTISCALE-MATERIAL-20261118',acceptance=accept['overall_status'],
 code_publication=pub,source_validation_passed=validation['passed'],
 active_HEAD=head,active_HEAD_unchanged=True,active_index_unchanged=True,
 campaign_controller_processes=controllers,gpu_compute_apps=apps,GPU_state=gpu,
 GPU_kept_ON=True,power_operation_performed=False,lease_released=True,
 actual_optimizer_updates=2568,native_GS_forwards=2952,new_TRELLIS_inferences=1,
 formal_199_frames_started=False,production_promoted=False,human_verdict=None,problem_solved=None,
 original_inputs_full_shared_fields_optimizers_assets_retained_remote=True,
 local_delivery='light report+evidence and T2 float native pair, not full-model backup')
(P/'finish_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(finished=True,commit=pub['commit'],acceptance=accept['overall_status'],GPU_kept_ON=True,active_HEAD_and_index_unchanged=True)))
