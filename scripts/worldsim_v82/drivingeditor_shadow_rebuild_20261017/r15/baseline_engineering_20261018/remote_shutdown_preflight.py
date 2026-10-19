"""Final A/B-aware AutoDL preflight; --shutdown only after local delivery/Git."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,os,re,subprocess
BASE=Path('/root/autodl-tmp/drivingeditor_repro');E=BASE/'campaign/nearby_actors_20261015_v1/baseline_engineering_20261018'
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--commit',required=True);ap.add_argument('--shutdown',action='store_true');a=ap.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.commit)
 slot=(BASE/'campaign/gpu_slot.lock').open('a+');fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
 q=read(E/'parallel_queue_status.json');assert q['state']=='PARALLEL_A_B_FULL49_COMPLETE_AWAITING_REPORT_GIT'
 guard=read(E/'parallel_group_guard.json');assert guard['state']=='TRAINING_ENDED'
 assert read(E/'all_vehicle_train_queue_status.json')['state']=='ALL_VEHICLE30K_FULL49_COMPLETE_AWAITING_REPORT_GIT'
 claim=read(E/'all_actor_gpu_claim.json');assert not claim.get('active',False)
 active=Path(read(E/'direct_active_run.json')['output']);assert read(active/'status.json')['state']=='complete'
 node=read(E/'parallel_A_030000_durability_receipt.json');assert node['model']=='A' and node['step']==30000
 assert (active/'checkpoint_030000.pt').stat().st_size==node['components'][0]['bytes']
 B=BASE/'campaign/r15_baseline_allvehicles_20261018/runs/scene0008_089_137_hybrid_30000_seed0_GN11';assert read(B/'status.json')['status']=='complete'
 event=read(B/'checkpoint_events/iteration_030000.json');assert event['status']=='complete'
 for v in event['checkpoints'].values():assert Path(v['path']).stat().st_size==v['bytes']
 ar=read(E/'direct_full49/receipt.json');br=read(E/'all_vehicle_full49/receipt.json')
 assert ar['frames']==br['frames']==49 and len(ar['records'])==245 and len(br['records'])==49 and ar['parameter_updates']==br['parameter_updates']==0
 assert read(E/'direct_full49/restore_receipt.json')['runtime_restoration_verified']
 package=read(E/'final_full49_delivery.json');assert package['status']=='COMPLETE_FINAL_FULL49_PNG_JSON_ARCHIVE'
 assert Path(package['archive']).stat().st_size==package['bytes']
 processes=[];blocked=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   status=(proc/'status').read_text();state=next(x for x in status.splitlines() if x.startswith('State:'))
  except (OSError,StopIteration):continue
  if not cmd:continue
  if any(x in cmd for x in ['python','ffmpeg','controller','queue']):processes.append(dict(pid=int(proc.name),state=state,executable=cmd.split()[0],arguments_not_published=True))
  if str(BASE) in cmd and any(x in cmd.lower() for x in ['python','ffmpeg','worker','controller','queue']):blocked.append(dict(pid=int(proc.name),executable=cmd.split()[0]))
 assert not blocked,blocked
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip();assert not gpu,gpu
 for pid in [79622,79624,79625,252862,252851]:
  proc=Path('/proc')/str(pid)
  if proc.exists():
   cmd=(proc/'cmdline').read_bytes().decode(errors='replace');assert str(E) not in cmd and 'audit_remaining_disk.py' not in cmd,dict(pid=pid,campaign_process_still_present=True)
 program=Path('/usr/bin/shutdown');assert program.is_file() and os.access(program,os.X_OK)
 receipt=dict(status='PASSED_CURRENT_A_B_ALL_WORK_COMPLETE_SHUTDOWN_PREFLIGHT',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),delivery_git_commit=a.commit,
  queue=q,guard=guard,A_step=30000,B_step=30000,A_records=245,B_records=49,archive_bytes=package['bytes'],archive_sha256=package['sha256'],
  active_campaign_workers=[],gpu_compute_processes=gpu,observed_python_service_processes=processes,all_actor_gpu_claim_active=False,
  shutdown_entry=str(program),shutdown_entry_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),shutdown_requested=a.shutdown,cloud_billing_verified=False)
 if a.shutdown:(E/'shutdown_preflight_current.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt),flush=True)
 if a.shutdown:
  os.sync()
  # The verified AutoDL entry is a shell script without a shebang.
  subprocess.run(['/bin/bash',str(program)],check=True)
 slot.close()
if __name__=='__main__':main()
