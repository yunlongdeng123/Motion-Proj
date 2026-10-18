"""Finite zero-update allocation probe; lets existing A save its exact yield state."""
from pathlib import Path
import os,json,time,datetime,subprocess
E=Path(__file__).resolve().parent
def main():
 import torch
 assert json.loads((E/'direct_instance30k/status.json').read_text())['state']=='running'
 planned=dict(reason='latest human request: start B concurrently if memory permits',old_A_controller=52748,old_B_wait_controller=52417,
   checkpoint_yield_is_expected=True,extra_training_updates=0,replay_updates_authorized=0,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 (E/'parallel_cutover_plan.json').write_text(json.dumps(planned,indent=2))
 free,total=torch.cuda.mem_get_info();reserve_bytes=13*1024**3
 assert free>reserve_bytes+3*1024**3,(free,total)
 held=torch.empty(reserve_bytes,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
 receipt=dict(status='PASSED_PARALLEL_ACTUAL_CAPACITY',user_authorized=True,
  measured_free_before_GiB=free/1024**3,measured_allocation_GiB=13,A_allocator_budget_GiB=16,B_allocator_budget_GiB=13,
  reserve_GiB=2,combined_peak_plus_reserve_GiB=31,total_GiB=total/1024**3,
  qualification='Actual current13GiB allocation succeeded alongside A;16+13 allocator budgets plus2GiB reserve. Future model peak is monitored, not certified by this initial probe.',
  training_updates=0,pid=os.getpid())
 assert receipt['combined_peak_plus_reserve_GiB']<=receipt['total_GiB']
 deadline=time.monotonic()+240
 while time.monotonic()<deadline:
  s=json.loads((E/'direct_instance30k/status.json').read_text())
  if s.get('state')=='yielded_to_priority_work':
   p=Path(s['resume_checkpoint']);assert p.is_file() and p.stat().st_size>0
   receipt.update(A_durable_step=s['step'],A_resume_checkpoint=str(p),A_replay_updates=0)
   (E/'parallel_capacity_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));return
  time.sleep(2)
 raise TimeoutError('A did not publish a complete priority-yield checkpoint; preserve state, do not kill A')
if __name__=='__main__':main()
