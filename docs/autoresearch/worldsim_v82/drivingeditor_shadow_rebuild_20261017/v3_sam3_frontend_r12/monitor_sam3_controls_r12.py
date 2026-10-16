"""Read actual finite-controller status; never calls an old E6 continuation."""
from pathlib import Path
import json,subprocess,time,statistics,shutil
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');root=R/'ownership_controls_r12_sam3';owned=[];training=False
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode();state=(proc/'stat').read_text().split()[2]
 except (OSError,UnicodeError):continue
 if state=='Z':continue
 if 'three_control_controller_r12.py' in cmd or (str(root) in cmd and any(q in cmd for q in ['run.py train','C_split_r12.py','C_finalize_owner_r12.py','r12_geometry_review.py','fullscene_ownership_audit_common_r9.py'])):
  owned.append(dict(pid=int(proc.name),state=state,cmd=cmd));training|='run.py train' in cmd
arms=[];speed=[];logged=0
for name in ['A_box','B_semantic_motion','C_correctable']:
 d=root/name;p=d/'progress.json';value=json.loads(p.read_text()) if p.exists() else {};step=value.get('global_step',0);logged+=step
 metrics=d/'metrics.jsonl';tail=[json.loads(x) for x in metrics.read_text().splitlines()[-8:]] if metrics.exists() else []
 for a,b in zip(tail,tail[1:]):
  delta=b['global_step']-a['global_step'];sec=b['elapsed_seconds']-a['elapsed_seconds']
  if 0<delta<=50 and sec>0:speed.append(sec/delta)
 arms.append(dict(arm=name,logged_step=step,complete=(d/'complete.json').exists(),progress=value if step else None))
seconds=statistics.median(speed) if speed else None
value=dict(checked_epoch=time.time(),status=json.loads((root/'pipeline_state.json').read_text()) if (root/'pipeline_state.json').exists() else None,controller_launch=json.loads((root/'controls_launch.json').read_text()),owned_live_processes=owned,arms=arms,recent_seconds_per_update=seconds,training_only_remaining_minutes=None if seconds is None else max(0,9000-logged)*seconds/60,ETA_excludes='full49x65 render/encoding, staged C evidence CPU checks, transfer/root visual review; no later material budget started',charged_completed_updates_range=[13500+logged,13549+logged+(49 if training else 0)],C_code_ready=(root/'C_correction_code_ready.json').exists(),GPU=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip(),disk_free_GiB=shutil.disk_usage(R).free/2**30)
(root/'last_monitor.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
