"""仅从保存的真实 native 数组核对残余薄覆盖；不导入 Torch/CUDA。"""
from pathlib import Path
import argparse
import json
import shutil
from datetime import datetime
import numpy as np
from road_activation_builder import CAMPAIGN, CPU_CAMPAIGN, PREVIOUS, read_npz, write_json


def diagnose(native_dir):
    native = Path(native_dir)
    road = read_npz(native/'asset_A1_guarded/road.npz')
    prior = read_npz(native/'asset_A1_guarded/provenance.npz')
    rows = []
    for raw in (86, 110):
        data = read_npz(native/f'{raw:03d}_native.npz')
        teacher = read_npz(CPU_CAMPAIGN/f'supervision_v4_candidate_r1/{raw:03d}.npz')
        alpha, before = data['road_alpha'], data['baseline_road_alpha']
        remaining = data['fixed_low_S'] & (alpha <= .9)
        ids = teacher['triangle_world_rows'][remaining]
        weights = teacher['triangle_weights'][remaining]
        use = weights > 1e-7
        def anyflag(value):
            return (value[ids] & use).any(1)
        def allflag(value):
            return (value[ids] | ~use).all(1)
        flags = dict(inactive_any_corner=anyflag(~road['active']),
            inactive_all_corners=allflag(~road['active']), unknown_any_corner=anyflag(prior['unknown']),
            legacy_legal_route_false_any=anyflag(~prior['legal_write']),
            oldactive_all_corners=allflag(road['active'] & ~prior['activation_new']),
            all_corners_active=allflag(road['active']), new_activation_any_corner=anyflag(prior['activation_new']),
            K_guard_any=anyflag(prior['activation_K_guard']), near_guard_any=anyflag(prior['activation_near_guard']))
        # 与主 native receipt 使用相同 float32 比较式，避免阈值相减次序引入计数差异。
        drop = data['S_full'] & (alpha < before - 1e-6)
        rows.append(dict(raw=raw, remaining_thin=int(remaining.sum()), remaining_gap=int((remaining & (alpha < .05)).sum()),
            alpha_min=float(alpha[remaining].min()), alpha_median=float(np.median(alpha[remaining])),
            alpha_max=float(alpha[remaining].max()), triangle_overlapping_flags={key:int(value.sum()) for key,value in flags.items()},
            observed_high_alpha_drop=dict(pixels=int(drop.sum()), max_drop=float((before[drop]-alpha[drop]).max()) if drop.any() else 0.,
                before_min=float(before[drop].min()) if drop.any() else None,
                after_min=float(alpha[drop].min()) if drop.any() else None,
                comparator='after < before - 1e-6 in saved float32 arrays',
                causal_explanation='not established; inherited native early termination may be relevant; no rounding-only claim',
                supported_to_thin_or_gap_pixels=int((drop & (before > .9) & (alpha <= .9)).sum()))))
        ys, xs = np.nonzero(remaining)
        path = native/f'{raw:03d}_remaining_thin.npz'
        if path.exists():
            shutil.copy2(path, path.with_name(path.name+'.before_neutral_label'))
        np.savez_compressed(path, pixel_y=ys, pixel_x=xs, triangle_world_rows=ids, triangle_weights=weights,
            alpha_before=before[remaining], alpha_after=alpha[remaining], **flags)
    baseline = []
    for raw in (86, 110, 134, 179, 198):
        data = read_npz(native/f'{raw:03d}_native.npz')
        old = read_npz(PREVIOUS/f'delivery/public_native_r1/baseline/{raw:03d}_native.npz')
        baseline.append(dict(raw=raw, RGB_max=float(np.abs(data['baseline_road_rgb']-old['road_rgb']).max()),
            alpha_max=float(np.abs(data['baseline_road_alpha']-old['road_alpha']).max())))
    result = dict(schema='road_activation_remaining_thin_cpu.v2', fixed_low_S=965,
        remaining_thin=sum(row['remaining_thin'] for row in rows), remaining_gap=sum(row['remaining_gap'] for row in rows),
        rows=rows, fresh_parent_road_baseline_vs_frozen=baseline, CPU_only=True, additional_native_calls=0,
        optimizer_updates=0,
        boundary='triangle properties are source-state association, not per-row alphaT attribution or evidence for expanding the activation neighborhood')
    path = native/'remaining_thin_cpu.json'
    if path.exists():
        stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        shutil.copy2(path, path.with_name(path.name+'.before_neutral_label.'+stamp))
    write_json(path, result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=CAMPAIGN/'native_r1')
    print(json.dumps(diagnose(parser.parse_args().native), ensure_ascii=False, allow_nan=False))
