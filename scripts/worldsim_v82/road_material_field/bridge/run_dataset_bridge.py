"""由通用样本清单与模型输出清单运行固定表面的材质光栅对照。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--sample', required=True)
    parser.add_argument('--texture', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raster', type=Path, required=True)
    args = parser.parse_args()
    args.dataset=args.dataset.resolve(); args.output=args.output.resolve(); args.raster=args.raster.resolve()
    args.texture=[path.resolve() for path in args.texture]
    dataset = json.loads(args.dataset.read_text())
    rows = [x for x in dataset['samples'] if x['id'] == args.sample]
    if len(rows) != 1:
        raise ValueError('未找到唯一的已登记样本')
    sample = rows[0]
    matrix = args.output.parent / (args.output.name + '_matrix.json')
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_text(json.dumps(sample['world_from_uv']))
    entry = [sys.executable,'-m',__package__+'.render_field'] if __package__ else [sys.executable,str(Path(__file__).with_name('render_field.py'))]
    command = entry + [
               '--matrix', str(matrix), '--support', str(args.dataset.parent / sample['valid_mask']),
               '--world-xyz', str(args.dataset.parent / sample['world_xyz']),
               '--hole-mask', str(args.dataset.parent / sample['hole_mask']),
               '--uv-corner', '--raster', str(args.raster), '--output', str(args.output)]
    for texture in args.texture:
        command += ['--texture', str(texture)]
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
