from pathlib import Path
import os,json,fcntl,ast
import torch
E=Path(__file__).resolve().parent
def main():
 lock=Path('/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((E/'all_actor_gpu_claim.json').read_text())['active'] is False
 s=(E/'gn11_train_timing_safe.py').read_text();ast.parse(s);assert "iter_end.synchronize()\n            log_dict['iter_time'] = iter_start.elapsed_time(iter_end)" in s
 vals=[]
 for _ in range(20):
  start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);a=torch.randn((128,128),device='cuda',requires_grad=True);start.record();(a@a).square().mean().backward();end.record();end.synchronize();vals.append(start.elapsed_time(end))
 assert all(v>=0 for v in vals)
 out=dict(status='PASSED_CUDA_EVENT_COMPLETION_TIMING',cases=20,training_updates=0,native_model_GPU_restore_verified=False,fix='recorded end event synchronize before elapsed_time query; loss/optimizer/geometry unchanged',elapsed_ms=vals)
 (E/'GN11_timing_fix_GPU_receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
