"""CPU 只读：严格核对模型/Adam/RNG/采样；只排除明确的执行路径和道路配置。"""
import argparse
import json
from pathlib import Path
import torch
from compare_v1_states import differences


def compare(reference, candidate, iteration, include_road=True):
    errors = []
    for prefix in ('chkpnt', 'dynamic_chkpnt', 'env_light_chkpnt', 'training_state'):
        left, right = [torch.load(p/f'{prefix}{iteration}.pth', map_location='cpu', weights_only=False)
                       for p in (reference, candidate)]
        if prefix == 'training_state':
            for state in (left, right):
                state['config'] = {k:v for k,v in state['config'].items()
                    if k not in ('model_path', 'start_checkpoint', 'v4_arm', 'v4_bundle', 'v41_dense_manifest')}
        errors.extend(differences(left, right, prefix))
        del left, right
    if include_road:
        left, right = [torch.load(p/f'road_chkpnt{iteration}.pth', map_location='cpu', weights_only=False)
                       for p in (reference, candidate)]
        for state in (left, right): state.pop('cumulative_seconds')
        errors.extend(differences(left, right, 'road'))
    return dict(exact_equal=not errors, differences=errors, iteration=iteration,
        reference=str(reference), candidate=str(candidate), include_road=include_road,
        excluded_native_config=['model_path', 'start_checkpoint', 'v4_arm', 'v4_bundle', 'v41_dense_manifest'],
        excluded_road_state=['cumulative_seconds'] if include_road else [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True); p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True); p.add_argument('--native-only', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.reference, args.candidate, args.iteration, not args.native_only)
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not result['exact_equal']: raise SystemExit(1)


if __name__ == '__main__': main()
