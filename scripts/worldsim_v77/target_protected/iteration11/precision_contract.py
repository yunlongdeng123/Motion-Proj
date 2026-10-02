"""混合精度加载：在整体降精度前保留实际可训练参数的原始FP32值。"""
import torch

def cast_frozen_preserve_trainable(model, names, device, frozen_dtype=torch.bfloat16):
    parameters = dict(model.named_parameters())
    if not names or len(set(names)) != len(names) or any(k not in parameters for k in names):
        raise ValueError('可训练参数列表为空、重复或不属于模型')
    # 必须在整体cast前复制；BF16再转FP32无法恢复原始权重。
    original = {k: parameters[k].detach().to(device='cpu', dtype=torch.float32).clone() for k in names}
    model.to(device=device, dtype=frozen_dtype)
    model.requires_grad_(False)
    parameters = dict(model.named_parameters())
    for k in names:
        parameters[k].data = original[k].to(device=device)
        parameters[k].requires_grad_(True)
    model.trainable_initialization_evidence = {
        'captured_before_frozen_dtype_cast': True,
        'trainable_dtype': 'torch.float32',
        'trainable_tensors': len(names),
        'frozen_dtype': str(frozen_dtype),
        'zero_step_trainable_rounding': False,
    }
    return [parameters[k] for k in names]
