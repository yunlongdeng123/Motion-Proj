"""Use the authorized AutoDL entry only after verified local/Git delivery."""
from pathlib import Path
import datetime,json,subprocess,time
import publish_checkpoint_milestone as p
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 plan=json.loads((p.O/'direct_plan.json').read_text(encoding='utf-8'));assert plan['report']['phase']['complete']
 review=json.loads((p.O/'final_quality_review.json').read_text(encoding='utf-8'));assert review['status']=='COMPLETED_ACTUAL_FULLFRAME_QUALITY_REVIEW'
 assert set(json.loads((p.O/'results_status.json').read_text())['completed_models'])=={'gn11five','matched5','allvehicles'}
 assert json.loads((p.O/'final_delivery_collection.json').read_text())['all_member_hashes_verified']
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.P,text=True).strip()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/research/worldsim-v8.2-drivingeditor-debug'],cwd=p.P,text=True).split()[0];assert head==remote
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p.P).strip()
 for name in ['final_quality_review.json','direct_full49/receipt.json','all_vehicle_full49/receipt.json','parallel_A_030000_durability_receipt.json']:
  subprocess.run(['git','cat-file','-e',head+':'+(p.D/name).as_posix()],cwd=p.P,check=True)
 source=(p.W/'remote_shutdown_preflight.py').read_text(encoding='utf-8');started=utc()
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','wm-vgpu-1011','/usr/bin/python3','-','--commit',head,'--shutdown'],input=source,text=True,capture_output=True,timeout=90)
 (p.W/'shutdown_execution.log').write_text(result.stdout+'\nSTDERR\n'+result.stderr,encoding='utf-8')
 rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')];assert rows,('no preflight acknowledgement',result.returncode,result.stderr)
 preflight=rows[0];assert preflight['status']=='PASSED_CURRENT_A_B_ALL_WORK_COMPLETE_SHUTDOWN_PREFLIGHT' and preflight['shutdown_requested']
 for base in [p.W,p.O]:(base/'shutdown_preflight_current.json').write_text(json.dumps(preflight,indent=2))
 observations=[];failures=0
 for n in range(7):
  check=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','wm-vgpu-1011','true'],text=True,capture_output=True,timeout=12)
  reachable=check.returncode==0;observations.append(dict(at=utc(),SSH_reachable=reachable,returncode=check.returncode,detail=check.stderr[-1000:]))
  print(json.dumps(observations[-1]),flush=True)
  failures=0 if reachable else failures+1
  if failures>=3:break
  time.sleep(10)
 receipt=dict(status='SHUTDOWN_REQUESTED_SSH_UNREACHABLE_OBSERVED' if failures>=3 else 'SHUTDOWN_REQUESTED_SSH_STILL_REACHABLE_UNCONFIRMED',request_started_at=started,completed_at=utc(),delivery_git_commit=head,ssh_shutdown_returncode=result.returncode,entry='/usr/bin/shutdown',observations=observations,cloud_billing_verified=False)
 for base in [p.W,p.O]:(base/'shutdown_status_current.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt),flush=True)
 assert failures>=3,'shutdown not confirmed by SSH observation; do not claim success'
if __name__=='__main__':main()
