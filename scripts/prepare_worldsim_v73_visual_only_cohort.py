"""仅按既有build元数据组织零LiDAR子集，不复制测量或读取评价结果。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tick = time.monotonic()
    parent = args.actor_data.resolve()
    original = json.loads((parent / 'index.json').read_text())
    cases = [row for row in original['cases'] if row['build_points'] == 0]
    args.output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + '\n')

    boundary = ('All original zero-build-LiDAR Actors selected by existing build metadata; '
                'camera-absent Actors retained; no selection by target availability, '
                'predicted native support or model quality. Original case files linked unchanged.')
    save('manifest.json', {'parent_data': str(parent), 'selection': boundary,
         'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                text=True).strip(),
         'failure_ledger_refs': ['V73-F02', 'V73-F03', 'V73-F05'],
         'optimizer_updates': 0, 'external_data_read': False})
    save('status.json', {'status': 'running'})
    for row in cases:
        destination = args.output / row['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(parent / row['file'])
    save('index.json', {'cases': cases, 'selection_boundary': boundary, 'parent_data': str(parent)})
    summary = {'status': 'done', 'actors': len(cases),
               'roles': dict(Counter(row['role'] for row in cases)),
               'logs_by_role': {role: len({row['log_id'] for row in cases if row['role'] == role})
                                for role in sorted({row['role'] for row in cases})},
               'original_input_status': dict(Counter(row['status'] for row in cases)),
               'wall_s': time.monotonic() - tick, 'boundary': boundary,
               'failure_ledger_delta': 'none; input organization only, no training result'}
    save('summary.json', summary)
    save('status.json', {'status': 'done'})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
