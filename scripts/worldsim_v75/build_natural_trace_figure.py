"""用真实 RGB、已记录三维状态和实际条件图展示输入链条。"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


def build_trace(trace, output):
    a = argparse.Namespace(trace=trace, output=output)
    a.output.mkdir(parents=True, exist_ok=True)
    data = json.loads((a.trace / 'trace_data.json').read_text(encoding='utf-8'))
    states = {r['variant']: r for r in data['states']}
    colors = {'gt_clean': '#344558', 'dvgt_metric': '#cd403b',
              'ordinary_bbox': '#8163b3', 'reference_lidar': '#118271'}
    names = {'gt_clean': 'GT annotation', 'dvgt_metric': 'DVGT + known rays',
             'ordinary_bbox': 'Ordinary box fit', 'reference_lidar': 'Extra target LiDAR'}
    fig = plt.figure(figsize=(16, 6.3), layout='constrained')
    grid = fig.add_gridspec(3, 3, width_ratios=[1.25, 1, 1.1])
    rgb = Image.open(a.trace / 'initial_rgb.png')
    ax = fig.add_subplot(grid[:2, 0])
    ax.imshow(rgb)
    x0, y0, x1, y1 = data['initial_reference_box']
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#ffdb28', linewidth=2))
    ax.set_title('(a) Real initial RGB (0 s)', loc='left', fontsize=13)
    ax.axis('off')
    ax = fig.add_subplot(grid[2, 0])
    # 所有状态共享这一个实际初始画面；裁剪仅便于看到目标。
    ax.imshow(rgb.crop((x0-75, y0-35, x1+75, y1+35)))
    ax.set_title('Same white car; same RGB in every arm', fontsize=10)
    ax.axis('off')
    ax = fig.add_subplot(grid[:, 1])
    for name in names:
        row = states[name]
        p = row['bev_right_forward']
        ax.add_patch(Polygon(row['bev_hull'], closed=True, fill=False,
                             edgecolor=colors[name], linewidth=1.8,
                             linestyle='--' if name == 'ordinary_bbox' else '-', label=names[name]))
        ax.scatter(*p, s=22, color=colors[name], zorder=5)
    low = states['dvgt_metric']['bev_right_forward'][1]
    high = states['gt_clean']['bev_right_forward'][1]
    ax.annotate('', xy=(2.6, low), xytext=(2.6, high),
                arrowprops={'arrowstyle': '<->', 'color': colors['dvgt_metric'], 'lw': 1.7})
    ax.text(2.35, (low+high)/2, f'{high-low:.2f} m\ncloser', ha='right', va='center',
            color=colors['dvgt_metric'], fontsize=10)
    ax.set_xlim(0.6, 8.5); ax.set_ylim(30, 43.5); ax.set_aspect('equal')
    ax.set_title('(b) Initial 3D actor state', loc='left', fontsize=13)
    ax.set_xlabel('Ego right (m)'); ax.set_ylabel('Ego forward (m)')
    ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper left', framealpha=.95)
    ax.text(.5, -.14, 'Ego origin (0, 0) is below this zoom.\nShared GT size / yaw; translation changes.',
            ha='center', va='top', transform=ax.transAxes, fontsize=9)
    for i, name in enumerate(['gt_clean', 'dvgt_metric', 'reference_lidar']):
        ax = fig.add_subplot(grid[i, 2])
        ax.imshow(Image.open(a.trace / f'condition-{name}.png'))
        ax.set_title(('(c) Actual condition crops (1 s)\n' if i == 0 else '') + names[name],
                     loc='left', fontsize=13 if i == 0 else 10, color=colors[name])
        ax.axis('off')
    fig.savefig(a.output / 'input-state-condition.png', dpi=180)
    fig.savefig(a.output / 'input-state-condition.svg')
    plt.close(fig)
    svg = a.output / 'input-state-condition.svg'
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build_trace(args.trace, args.output)
