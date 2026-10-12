"""只读小型状态/显存摘要，不导入torch、不加载模型。"""
import argparse
import datetime
import json
from pathlib import Path

from run_matrix import ROOT


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def snapshot():
    result = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'root': str(ROOT), 'matrix': read(ROOT / 'matrix_status.json'), 'arms': {}}
    for arm in ('C00', 'G10', 'N01', 'GN11'):
        cfg = read(ROOT / 'configs' / (arm + '.json'))
        output = Path(cfg['model_path'])
        status = read(output / 'status.json')
        memory = read(output / 'memory_summary.json')
        row = {'output': str(output), 'status': status['status'] if status else 'not_started',
               'iteration': status.get('iteration', 0) if status else 0,
               'pid': status.get('pid') if status else None,
               'wall_seconds': status.get('wall_seconds') if status else None,
               'speed_window': status.get('speed_window') if status else None,
               'loss': status.get('loss') if status else None,
               'psnr': status.get('psnr') if status else None,
               'error': status.get('error') if status else None,
               'complete_checkpoint_nodes': []}
        for step in (10000, 20000, 30000):
            event = read(output / 'checkpoint_events' / ('iteration_%06d.json' % step))
            if event and event.get('status') == 'complete':
                row['complete_checkpoint_nodes'].append(step)
        if memory:
            row['memory'] = {key: memory[key] for key in ('pytorch', 'nvidia_smi', 'sampling',
                           'phases', 'rtx3090_assessment', 'error_count', 'extra', 'run_status')}
        result['arms'][arm] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    cli = parser.parse_args()
    record = snapshot()
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    if cli.output:
        cli.output.write_text(payload, encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
