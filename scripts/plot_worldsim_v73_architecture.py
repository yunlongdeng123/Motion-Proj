"""绘制当前V7.3的实际组件与只读几何数据流，不预写未实现模块。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16, 5.1))
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
    ax.set(xlim=(0, 16), ylim=(0, 5.1))
    ax.axis('off')
    ink = '#25313b'
    colors = {'input': '#fff3cd', 'frozen': '#deebf7',
              'trainable': '#dff0df', 'given': '#f2f3f5', 'output': '#fce4d6'}

    def box(x, y, w, h, text, kind, fontsize=20):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.035,rounding_size=.12',
                                   facecolor=colors[kind], edgecolor=ink, linewidth=1.8))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize,
                color=ink, linespacing=1.2)

    def arrow(points):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=ink, lw=1.8)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                    mutation_scale=19, linewidth=1.8, color=ink))

    # 主行只画构建输入到规范表面的实际可学习通路。
    y, h = 3.0, 1.3
    nodes = [(.12, 1.42, 'Multi-view\nimages', 'input'),
             (1.94, 1.97, 'Shared VGGT\nprefix', 'frozen'),
             (4.31, 1.80, 'Native DPT\ndecoder', 'trainable'),
             (6.51, 2.05, 'Depth +\n4-scale maps', 'given'),
             (8.96, 2.46, '3D queries\nlocal interaction\n+ view sampling', 'trainable'),
             (11.82, 1.73, 'Patch\nhead', 'trainable'),
             (13.95, 1.91, 'Canonical\nActor surface', 'output')]
    for x, w, label, kind in nodes:
        box(x, y, w, h, label, kind, fontsize=18 if x == 8.96 else 20)
    for (x, w, _, _), (next_x, _, _, _) in zip(nodes, nodes[1:]):
        arrow([(x+w+.04, y+h/2), (next_x-.04, y+h/2)])
    ax.text(10.19, 4.52, r'$\times 3$ updates', ha='center', fontsize=18, color=ink)

    box(.12, 1.05, 2.85, 1.05, 'Sparse build\nLiDAR observations', 'input', 20)
    arrow([(2.97, 1.58), (3.37, 1.58), (3.37, 2.47), (9.56, 2.47), (9.56, 2.96)])
    # 深度种子与多尺度读取都来自当前DPT，不从旧最终特征缓存读取。
    arrow([(7.54, 2.96), (7.54, 2.47)])
    ax.text(5.3, 2.59, 'Canonical seeds', ha='center', fontsize=16, color=ink)
    box(4.03, .94, 4.58, 1.16, 'Calibration + metric scale\n+ known Actor trajectories', 'given', 19)
    arrow([(8.65, 1.52), (10.76, 1.52), (10.76, 2.96)])
    box(11.82, .94, 4.04, 1.16, 'Known motion + background\nHard first-intersection readout', 'given', 18)
    arrow([(14.90, 2.96), (14.90, 2.14)])

    for x, kind, label in [(0.2, 'frozen', 'Frozen'), (2.25, 'trainable', 'Trainable'),
                            (4.70, 'given', 'Fixed operation / read-only data')]:
        ax.add_patch(FancyBboxPatch((x, 4.76), .23, .23, boxstyle='round,pad=.01,rounding_size=.03',
                                   facecolor=colors[kind], edgecolor=ink, linewidth=1))
        ax.text(x+.35, 4.87, label, va='center', fontsize=16, color=ink)
    ax.text(.12, .31,
            'Native control: DPT depth → canonical LiDAR fusion → PCA patches → the same physical readout.',
            fontsize=18, color=ink, va='center')
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(args.output.with_suffix('.'+extension), dpi=200, facecolor='white')
    svg = args.output.with_suffix('.svg')
    svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
    plt.close(fig)


if __name__ == '__main__':
    main()
