"""CPU离线汇总真实监督诊断；没有GPU文件时明确未执行，不填零。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def export(run, output):
    output.mkdir(parents=True, exist_ok=False)
    records, samples = [], []
    for metadata_path in sorted((run/'supervision_audit').glob('*/metadata.json')):
        record = json.loads(metadata_path.read_text(encoding='utf-8'))
        record['source'] = str(metadata_path)
        records.append(record)
        scale = record['meter_to_model']
        with np.load(metadata_path.with_name('maps.npz'), allow_pickle=False) as data:
            for index, (yy, xx) in enumerate(data['sample_pixels_yx']):
                samples.append(dict(frame=record['frame'], iteration=record['iteration'],
                    completed_optimizer_updates=record['completed_optimizer_updates'], y=int(yy), x=int(xx),
                    lidar_depth_m=float(data['sample_measured_depth'][index]/scale),
                    static_mean_depth_m=float(data['sample_static_mean_depth'][index]/scale),
                    mean_depth_front=bool(data['sample_mean_front'][index]),
                    **{key: float(data[key][index]) for key in [
                        'static_front_alpha', 'static_front_center_inside_alpha', 'static_front_road_alpha',
                        'static_center_inside_alpha', 'joint_front_static_alpha']}))
    summary = dict(status='EXPORTED_REAL_DIAGNOSTICS' if records else 'NOT_EXECUTED_NO_DIAGNOSTIC_FILES',
                   run=str(run), records=records, sampled_pixels=len(samples),
                   CUDA_executed_by_this_tool=False, optimizer_updates_by_this_tool=0,
                   dense_front_metric=False, quality_accepted=False, human_verdict=None)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if samples:
        with (output/'samples.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(samples)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.run, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}))
