"""验收第一项：比较 B/R/RP 的原 V1 参数、optimizer、RNG 与采样状态。"""

from pathlib import Path
import argparse
import json
import numpy as np
import torch


def differences(left, right, path='root'):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape or left.dtype != right.dtype:
            return [path+': tensor schema differs']
        return [] if torch.equal(left.cpu(), right.cpu()) else [path+': tensor values differ']
    if isinstance(left, np.ndarray):
        return [] if isinstance(right, np.ndarray) and np.array_equal(left, right) else [path+': array differs']
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            return [path+': dictionary keys differ']
        return [error for key in left for error in differences(left[key], right[key], f'{path}.{key}')]
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return [path+': sequence schema differs']
        return [error for i, (a, b) in enumerate(zip(left, right)) for error in differences(a, b, f'{path}[{i}]')]
    return [] if left == right else [path+': value differs']


def compare(reference, candidate, iteration):
    errors = []
    for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt'):
        # 每次只载入一个组件对，避免同时持有三组大模型。
        left = torch.load(reference/f'{prefix}{iteration}.pth', map_location='cpu')
        right = torch.load(candidate/f'{prefix}{iteration}.pth', map_location='cpu')
        errors.extend(differences(left, right, prefix))
        del left, right
    left = torch.load(reference/f'training_state{iteration}.pth', map_location='cpu')
    right = torch.load(candidate/f'training_state{iteration}.pth', map_location='cpu')
    for state in (left, right):
        config = state['config']
        state['config'] = {key: value for key, value in config.items()
                           if key != 'model_path' and not key.startswith('v4_')}
    errors.extend(differences(left, right, 'training_state'))
    return dict(iteration=iteration, reference=str(reference), candidate=str(candidate),
                exact_equal=not errors, differences=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    result = compare(options.reference, options.candidate, options.iteration)
    options.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['exact_equal']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
