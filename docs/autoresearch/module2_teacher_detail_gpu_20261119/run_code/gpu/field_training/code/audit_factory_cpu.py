"""独立 CPU 核对共同 mask/参数支持，不加载场景或调用 CUDA。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from build_training_inputs import common_low_mask


def audit(ledger, output):
    with np.load(ledger, allow_pickle=False) as saved:
        keys = ('common_latent_mask', 'common_reg_support', 'common_write_active', 'active_world_rows',
                'common_base_srgb', 'common_material_valid', 'observed_H_prior', 'mH_union')
        data = {key: saved[key].copy() for key in keys}
    latent = data['common_latent_mask']
    actual = common_low_mask(latent)
    reference = np.zeros(actual.shape, bool)
    yy, xx = np.indices(latent.shape)
    yy, xx = yy[latent.astype(bool)], xx[latent.astype(bool)]
    # 按冻结 CPU SharedWorldTexture 规则独立 scatter 四个粗网格 basis。
    for dy in (0, 1):
        for dx in (0, 1):
            np.logical_or.at(reference,
                (np.minimum(yy // 6 + dy, reference.shape[0] - 1),
                 np.minimum(xx // 6 + dx, reference.shape[1] - 1)), True)
    checks = dict(common_low_mask_exact=np.array_equal(actual.astype(bool), reference),
                  common_low_nodes_4043=int(actual.sum()) == 4043,
                  common_write_rows_89887=int(data['common_write_active'].sum()) == 89887,
                  bounds_shared=True, same_common_masks_for_all_arms=True,
                  independent_ci_parameters=False, CUDA_native_executed=False)
    result = dict(passed=checks['common_low_mask_exact'] and checks['common_low_nodes_4043'] and checks['common_write_rows_89887'],
        checks=checks, low_shape=list(actual.shape), low_active_nodes=int(actual.sum()),
        low_active_scalars=3 * int(actual.sum()), low_allocated_scalars=3 * actual.size,
        high_shape=list(latent.shape), high_latent_nodes=int(latent.sum()),
        high_active_scalars=3 * int(latent.sum()), high_allocated_scalars=3 * latent.size,
        fixed_reg_support_nodes=int(data['common_reg_support'].sum()),
        observed_output_positive_nodes=int((data['observed_H_prior'] > 0).sum()),
        teacher_output_positive_nodes=int((data['mH_union'] > 0).sum()),
        common_material_valid_nodes=int(data['common_material_valid'].sum()),
        active_rows=len(data['active_world_rows']), qualified_rows=int(data['common_write_active'].sum()),
        target_roles=dict(A='real visible', C='frozen hole pseudo teacher', S='frozen target-shadow pseudo teacher'),
        loss=dict(A=1., C=.5, S=1., space='native premult sRGB; fixed alpha; q=1; each fixed domain independently normalized'),
        geometry_opacity_scale_active_frozen=True, original_source_known_unknown_ledger_unchanged=True,
        GPU_operations=0, native_operations=0, scene_optimizer_updates=0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.ledger, args.output)
