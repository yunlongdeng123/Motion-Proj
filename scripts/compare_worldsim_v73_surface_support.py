"""复用已保存R11计数，与新完成R10/R7的固定表面诊断做日志配对。"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measured', type=Path, required=True)
    parser.add_argument('--reused', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measured = json.loads(args.measured.read_text())
    reused = json.loads(args.reused.read_text())
    statistics = {name: measured['statistics'][name] for name in ['joint_r10', 'lidar_r7']}
    statistics['native_r11'] = reused['statistics']['native_r11']
    for stats in statistics.values():
        for row in [stats['equal_log_means'], *stats['per_log'].values()]:
            # 正确沿束交点蕴含测量邻近；二者差值揭示仅有邻近覆盖的部分。
            row['near_without_hit_band'] = row['near_surface'] - row['any_hit_band']
        stats['raw_counts']['near_without_hit_band'] = (
            stats['raw_counts']['near_surface'] - stats['raw_counts']['any_hit_band'])
    paired = {}
    current = statistics['joint_r10']['per_log']
    for name in ['lidar_r7', 'native_r11']:
        reference = statistics[name]['per_log']
        common = sorted(current.keys() & reference.keys())
        paired[name] = {}
        for metric in current[common[0]]:
            deltas = {key: current[key][metric] - reference[key][metric] for key in common}
            paired[name][metric] = {
                'logs': len(common), 'mean_delta': float(np.mean(list(deltas.values()))),
                'higher_logs': sum(value > 0 for value in deltas.values()),
                'lower_logs': sum(value < 0 for value in deltas.values()),
                'equal_logs': sum(value == 0 for value in deltas.values()), 'per_log_delta': deltas,
            }
    result = {
        'task': 'WS-V73-M2-SURFACE-SUPPORT-01', 'status': 'done',
        'sources': {'joint_r10': str(args.measured), 'lidar_r7': str(args.measured), 'native_r11': str(args.reused)},
        'statistics': statistics, 'paired_joint_r10_minus': paired,
        'protocol': 'same original cohort, full_track FIT labels and hard-range free semantics; model and optimizer costs differ; no new bootstrap or neural inference; R11 counts reused unchanged',
        'boundary': 'nearby surface is not correct along-ray support; a later correct intersection never replaces the physical first return; layer counts are numerical depths, not topology',
        'failure_ledger_delta': 'update V73-F02 support evidence; no new failure ID; triangle pending',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'methods': list(statistics), 'paired': paired}))


if __name__ == '__main__':
    main()
