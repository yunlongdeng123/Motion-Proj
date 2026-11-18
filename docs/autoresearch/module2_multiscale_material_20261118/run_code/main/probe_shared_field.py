"""真实共享field的0更新原生前反向与单step耗时；不创建正式队列。"""
from pathlib import Path
import argparse,json,time
import torch
from train_world_material import TrainingState,P

def run(output,owner):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    s=TrainingState(P/'cpu_exact_r1/field_initial.pt',P/'method_scope_r1/scope.npz',P/'stage0_native_r1',P/'execution_contract.json',owner)
    s.model.set_stage('C');torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    loss,terms=s.objective(86);grads=torch.autograd.grad(loss,[s.model.low,s.model.high]);torch.cuda.synchronize()
    result=dict(schema='multiscale_shared_field_native_gradient.v1',GPU_executed=True,optimizer_updates=0,
        photo_loss=float(loss),terms={k:float(v.detach()) for k,v in terms.items()},
        low_gradient=dict(nonzero=int((grads[0]!=0).sum()),finite=bool(torch.isfinite(grads[0]).all()),max_abs=float(grads[0].abs().max())),
        high_gradient=dict(nonzero=int((grads[1]!=0).sum()),finite=bool(torch.isfinite(grads[1]).all()),max_abs=float(grads[1].abs().max())),
        parameter_scope=s.mask_statistics,one_forward_backward_seconds=time.monotonic()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),qualified_rows=int(s.eligible.sum()),
        Sigma_and_scene_frozen=True,independent_ci_trainable=False,field_query='actual GH15 worldquad→exactchart→bilinear A0/L/mH*H→native raster')
    if not result['low_gradient']['nonzero'] or not result['high_gradient']['nonzero']:raise RuntimeError('真实共享field梯度不通')
    s.save_checkpoint(output/'initial_complete_state.pt');s.assert_frozen();s.ctx.close()
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-owner',required=True)
    a=p.parse_args();run(a.output,a.lease_owner)
