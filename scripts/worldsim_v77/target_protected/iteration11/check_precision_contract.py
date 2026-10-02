"""使用真实checkpoint的80张量验证初始化保真；只有CPU，不训练DriveEditor。"""
from pathlib import Path
import json
import torch
from safetensors import safe_open
from precision_contract import cast_frozen_preserve_trainable

def main():
    torch.set_num_threads(4)
    task = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
    base = '/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors'
    results = {}
    for run, folder in [('r7', task/'r7/encoder_fixed_lowres/training'), ('r14', task/'r14/training')]:
        keys = json.loads((folder/'config.json').read_text())['trainable_tensors']
        model = torch.nn.Module()
        model.weights = torch.nn.ParameterList()
        model.frozen_probe = torch.nn.Parameter(torch.tensor([4.123456], dtype=torch.float32))
        with safe_open(base, framework='pt', device='cpu') as f:
            for k in keys:
                model.weights.append(torch.nn.Parameter(f.get_tensor(k).float()))
            names = [f'weights.{i}' for i in range(len(keys))]
            params = cast_frozen_preserve_trainable(model, names, 'cpu')
            assert len(params) == 80
            assert all(p.dtype == torch.float32 and p.requires_grad and torch.equal(p.detach(), f.get_tensor(k).float()) for p,k in zip(params,keys))
        assert model.frozen_probe.dtype == torch.bfloat16 and not model.frozen_probe.requires_grad
        results[run] = {'actual_checkpoint_tensors_preserved_exact': 80, 'selected_FP32': True, 'frozen_BF16': True}
        del model,params
    # 检查恢复的FP32参数仍可回传；不作任何DriveEditor优化或模型推理。
    tiny = torch.nn.Linear(2,1)
    with torch.no_grad():tiny.weight.copy_(torch.tensor([[4.123456, .001234567]]))
    original = tiny.weight.detach().clone()
    selected = cast_frozen_preserve_trainable(tiny, ['weight'], 'cpu')
    tiny.weight.square().sum().backward()
    assert torch.equal(tiny.weight.detach(), original) and torch.equal(selected[0].grad, 2*original)
    result = {'stage':'passed','actual_tensor_checks':results,'FP32_gradient_exact':True,'DriveEditor_optimizer_steps':0,'GPU_forwards':0}
    (task/'r19/precision_fix_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result)
if __name__=='__main__':main()
