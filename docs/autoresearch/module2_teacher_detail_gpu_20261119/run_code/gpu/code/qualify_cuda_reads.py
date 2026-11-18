"""Actual CUDA qualification of frozen physical material read operators; no optimizer."""
from pathlib import Path
import gc,json,sys,time,hashlib
import numpy as np
import torch

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0,str(R/'scripts/worldsim_v82'))
from fixed_material_read_operator import FixedReadOperator

out=Q/'gpu_20261119/cuda_read_qualification_r1'
out.mkdir(exist_ok=False)
assert json.loads((Q/'gpu_20261119/authorization.json').read_text())['GPU_authorized']
torch.set_num_threads(4)
rng=np.random.default_rng(101)
cases=[('avg',Q/'permission/avg_sparse_gh31_r1/operator','observed','teacher'),
       ('pixel086',Q/'pixel_render/read_operator_box_r1/086','observed_H','observed_plus_teacher_H')]
results=[]
for reader,root,*gates in cases:
    start=time.monotonic(); op=FixedReadOperator.load(root)
    # Small bounded fields stay away from clamp saturation while covering all real cache rows.
    l_np=rng.normal(0,.008,(*op.low_shape,3)).astype(np.float32)
    h_np=rng.normal(0,.012,(*op.high_shape,3)).astype(np.float32)
    loss_weights=rng.uniform(.1,1,(len(op.base),3)).astype(np.float32)
    for gate in gates:
        l_cpu=torch.tensor(l_np,requires_grad=True);h_cpu=torch.tensor(h_np,requires_grad=True)
        cpu=op.torch_query(l_cpu,h_cpu,gate)
        cpu_loss=(cpu*torch.from_numpy(loss_weights)).mean()
        cpu_loss.backward()
        cpu_grad_l=l_cpu.grad.numpy().copy();cpu_grad_h=h_cpu.grad.numpy().copy()
        l_gpu=torch.tensor(l_np,device='cuda',requires_grad=True);h_gpu=torch.tensor(h_np,device='cuda',requires_grad=True)
        t=time.monotonic(); gpu=op.torch_query(l_gpu,h_gpu,gate);torch.cuda.synchronize()
        first_forward=time.monotonic()-t
        (gpu*torch.as_tensor(loss_weights,device='cuda')).mean().backward();torch.cuda.synchronize()
        rgb_diff=float(np.max(np.abs(gpu.detach().cpu().numpy()-cpu.detach().numpy())))
        gl=float(np.max(np.abs(l_gpu.grad.cpu().numpy()-cpu_grad_l)))
        gh=float(np.max(np.abs(h_gpu.grad.cpu().numpy()-cpu_grad_h)))
        t=time.monotonic()
        for _ in range(3):
            l_gpu.grad=None;h_gpu.grad=None
            value=op.torch_query(l_gpu,h_gpu,gate)
            (value*torch.as_tensor(loss_weights,device='cuda')).mean().backward()
        torch.cuda.synchronize();elapsed=(time.monotonic()-t)/3
        r=dict(reader=reader,gate=gate,rows=len(op.base),nonzeros_H=op.KH[gate].nnz,
               RGB_CPU_CUDA_abs_max=rgb_diff,low_gradient_abs_max=gl,high_gradient_abs_max=gh,
               CUDA_H_gradient_nonzero=int((h_gpu.grad!=0).sum().item()),
               first_forward_seconds=first_forward,steady_forward_backward_seconds=elapsed,
               CUDA_peak_allocated_MB=torch.cuda.max_memory_allocated()/2**20,
               verified=rgb_diff<=1e-6 and gl<=1e-7 and gh<=1e-7)
        assert r['verified'],r
        results.append(r);print(json.dumps(r),flush=True)
        del l_gpu,h_gpu,gpu,value,l_cpu,h_cpu,cpu
    results[-1]['reader_total_seconds']=time.monotonic()-start
    op._torch_cache.clear();del op;gc.collect();torch.cuda.empty_cache()
receipt=dict(schema='actual_CUDA_fixed_read_qualification.v1',verified=all(r['verified'] for r in results),
    device=torch.cuda.get_device_name(),torch=torch.__version__,cases=results,
    optimizer_updates=0,native_ctx_loaded=False,source=__file__,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    limitation='Read operator only; native renderer, K protection and full state lifecycle are separate mandatory gates.')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
