"""绘制固定表面的邻近覆盖、沿束支持与真实首交点诊断。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    methods = ['joint_r5', 'lidar_r9', 'native_r11']
    labels = ['R5 joint', 'R9 LiDAR', 'R11 native']
    metrics = [summary['statistics'][name]['equal_log_means'] for name in methods]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={'width_ratios': [1, 1.25]})
    fig.subplots_adjust(left=.055, right=.99, top=.86, bottom=.31, wspace=.25)
    x = np.arange(3)
    groups = [('near_surface', 'Surface within 0.2 m', '#819dac'),
              ('any_hit_band', 'Any intersection within 0.2 m', '#b2c9a0'),
              ('first_hit', 'Literal first hit', '#2f7d4e')]
    for i, (key, label, color) in enumerate(groups):
        values = [item[key] * 100 for item in metrics]
        bars = axes[0].bar(x + (i-1)*.25, values, .24, label=label, color=color)
        axes[0].bar_label(bars, fmt='%.1f', fontsize=9, padding=2)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 115), yticks=np.arange(0,101,20), ylabel='Actor / log equal mean (%)',
                title='Nearby coverage does not imply a ray hit')
    axes[0].legend(loc='upper left', fontsize=9)
    categories = [
        ('first_hit', 'Correct first hit', '#2f7d4e'),
        ('early_with_later_hit', 'Early; later correct hit exists', '#d55e00'),
        ('early_without_hit_but_near', 'Early; no correct hit, surface nearby', '#ed9f35'),
        ('early_without_hit_or_near', 'Early; no correct hit or nearby surface', '#7d231d'),
        ('late_but_near', 'Late; surface nearby', '#5478b0'),
        ('late_without_near', 'Late; no nearby surface', '#a6b8d0'),
        ('missing_but_near', 'Missing; surface nearby', '#6c757d'),
        ('missing_without_near', 'Missing; no nearby surface', '#d5dadd'),
    ]
    left = np.zeros(3)
    handles = []
    for key, label, color in categories:
        values = np.array([item[key] * 100 for item in metrics])
        bars = axes[1].barh(x, values, left=left, color=color, label=label, edgecolor='white', linewidth=.4)
        handles.append(bars[0])
        for row, value in enumerate(values):
            if value >= 5:
                axes[1].text(left[row]+value/2, row, f'{value:.1f}', ha='center', va='center',
                             fontsize=9, color='white' if color in ['#2f7d4e','#d55e00','#5478b0','#6c757d'] else '#202020')
        left += values
    axes[1].set(yticks=x, yticklabels=labels, xlim=(0, 100), xlabel='All owned returns partitioned (%)',
                title='First-return outcomes, with later support separated')
    axes[1].invert_yaxis()
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].grid(axis='y', alpha=.15)
    fig.legend(handles, [item[1] for item in categories], loc='lower center', bbox_to_anchor=(.5,.065),
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle('V7.3 fixed surfaces: coverage, support and first intersections', fontsize=17, y=.98)
    fig.text(.5,.025, '75 development Actors / 5 logs; 11,886 owned ray occurrences; 23 Actors without owned returns retained.\n'
             'Different training conditions: descriptive diagnosis, not a matched ablation. No geometry changed.',
             ha='center', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(args.output.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
