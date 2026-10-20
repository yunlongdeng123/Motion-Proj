"""CPU汇总实际运行日志；准入与训练分开，不用无执行数据填0性能。"""

import argparse
import json
from pathlib import Path
import statistics


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def export(run, output):
    if output.exists():
        raise FileExistsError(output)
    cost, capacity = records(run/'v3_render_cost.jsonl'), records(run/'v3_capacity.jsonl')
    visibility = [json.loads(path.read_text()) for path in sorted((run/'road_visibility').glob('*/metadata.json'))]
    training = [row for row in cost if not row['gate'] and row['backend'] == 'actual_cuda_events']
    phases = {}
    for row in training:
        for key, value in row['spans'].items():
            if value['cuda_stream_elapsed_ms'] is not None:
                phases.setdefault(key, []).append(value['cuda_stream_elapsed_ms'])
    summary = dict(status='ACTUAL_LOGS_ONLY_NO_QUALITY_VERDICT' if cost or capacity or visibility else 'NOT_EXECUTED_NO_RENDER_AUDIT',
        source_run=str(run), training_profiled_iterations=len(training), gate_profiled_iterations=sum(row['gate'] for row in cost),
        CUDA_phase_sample_ms={key: dict(n=len(values), median=statistics.median(values), maximum=max(values)) for key, values in phases.items()},
        sample_medians_are_not_total_run_compute=True, inclusive_spans_must_not_be_summed=True,
        no_V1_speedup_or_memory_delta_without_matched_V1_profile=True,
        cost_records=cost, capacity_records=capacity, road_visibility_records=visibility,
        attenuation_is_not_illegal_occlusion_ground_truth=True, quality_accepted=False, human_verdict=None)
    output.mkdir(parents=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.run, args.output)
    print(json.dumps({key: result[key] for key in ['status', 'training_profiled_iterations', 'gate_profiled_iterations']}))
