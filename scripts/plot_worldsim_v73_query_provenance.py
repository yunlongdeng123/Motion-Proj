"""Plot fixed query-origin decomposition and every metadata-moving Actor.

No rerendering or neural inference. Ground-plane panels retain every stored ray;
line segments connect the measured and predicted endpoints of the SAME beam.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
data = json.loads((args.run / 'summary.json').read_text())
methods = ['joint_r5', 'lidar_r6']
labels = ['Joint r5', 'LiDAR r6']
sources = ['evidence', 'completion']
source_labels = ['Build-LiDAR origin', 'Completion origin']
colors = ['#367ca1', '#c77845']

fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.6))
for ax, (metric, title, unit, factor) in zip(axes, [
    ('free_contribution_m', 'Free-space intrusion', 'm / original near-box ray', 1),
    ('owned_early_contribution', 'Early-return contribution', '% of owned rays', 100),
    ('owned_hit_contribution', 'Hit contribution', '% of owned rays', 100),
    ('center_to_build_mean_m', 'Center to nearest build point', 'm (not displacement)', 1),
]):
    maximum = 0
    for j, source in enumerate(sources):
        values = [data['statistics'][m]['sources'][source] for m in methods]
        means = [v['equal_log_means'][metric] * factor for v in values]
        x = np.arange(2) + (j - .5) * .35
        ax.bar(x, means, width=.32, color=colors[j], alpha=.75, label=source_labels[j])
        for i, value in enumerate(values):
            per_log = [v[metric] * factor for v in value['per_log'].values()]
            ax.scatter(x[i] + np.linspace(-.075, .075, len(per_log)), per_log,
                       s=20, color=colors[j], edgecolor='white', linewidth=.5, zorder=3)
            ax.text(x[i], means[i], f'{means[i]:.1f}' if factor == 100 else f'{means[i]:.3f}',
                    ha='center', va='bottom', fontsize=8)
            maximum = max(maximum, max(per_log))
    ax.set_xticks([0, 1], labels)
    ax.set_title(title, loc='left', fontsize=10)
    ax.set_ylabel(unit, fontsize=9)
    ax.set_ylim(0, maximum * 1.18)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Where do fixed first-intersection errors originate?', y=.98, fontsize=14)
fig.text(.5, .90, '75 development Actors / 5 independent logs; bars = equal log means, dots = individual logs',
         ha='center', fontsize=10)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .07), ncol=2, frameon=False)
fig.text(.5, .025, 'Labels describe initialization; both query types share later learned updates. Distance beyond build support alone is not an error.',
         ha='center', fontsize=9)
fig.subplots_adjust(left=.055, right=.987, top=.80, bottom=.24, wspace=.39)
for ext in ['png', 'pdf']:
    fig.savefig(args.output / ('V73_QUERY_PROVENANCE.' + ext), dpi=170)
plt.close(fig)

moving = sorted([r for r in data['actors'][methods[0]]
                 if (r['actor'].get('translation_speed_mps') or 0) > 2],
                key=lambda r: r['actor']['owner'])
error_colors = np.array(['#c64d45', '#398461', '#526bb3'])
legend = [Line2D([], [], color='#9b9b9b', marker='.', linestyle='', label='Build LiDAR'),
          Line2D([], [], color='#252525', marker='x', linestyle='', label='Measured endpoint'),
          Line2D([], [], color=error_colors[0], marker='o', linestyle='', label='Early (>0.2 m)'),
          Line2D([], [], color=error_colors[1], marker='o', linestyle='', label='Hit (within 0.2 m)'),
          Line2D([], [], color=error_colors[2], marker='o', linestyle='', label='Late (>0.2 m)'),
          Line2D([], [], color='#555555', marker='o', linestyle='', label='Build-origin patch'),
          Line2D([], [], color='#555555', marker='^', linestyle='', label='Completion-origin patch')]
with PdfPages(args.output / 'V73_MOVING_QUERY_INTERSECTIONS.pdf') as pdf:
    for page, start in enumerate(range(0, len(moving), 3), 1):
        subset = moving[start:start + 3]
        fig, axes = plt.subplots(len(subset), 2, figsize=(11.7, 11), squeeze=False)
        for row_number, row in enumerate(subset):
            entry = row['actor']
            owner = entry['owner']
            records = [np.load(args.run / (method + '__' + owner + '_moving.npz')) for method in methods]
            size = records[0]['size_lwh_m']
            bounds = [np.array([[-size[0] / 2, -size[1] / 2], [size[0] / 2, size[1] / 2]])]
            for record in records:
                bounds += [record[name][:, :2] for name in ['build', 'measured', 'predicted'] if len(record[name])]
            all_points = np.concatenate(bounds)
            low, high = all_points.min(0), all_points.max(0)
            center = (low + high) / 2
            span = max(float(np.max(high - low)), 1.) * 1.12
            for column, (record, label) in enumerate(zip(records, labels)):
                ax = axes[row_number, column]
                measured, predicted, build = record['measured'], record['predicted'], record['build']
                mask = record['predicted_return_mask']
                delta = record['predicted_range_m'][mask] - record['observed_range_m'][mask]
                classes = np.where(delta < -.2, 0, np.where(delta > .2, 2, 1))
                if len(build):
                    ax.scatter(build[:, 0], build[:, 1], s=2, color='#a3a3a3', alpha=.4, rasterized=True)
                if len(measured):
                    ax.scatter(measured[:, 0], measured[:, 1], s=10, marker='x', linewidths=.4,
                               color='#252525', alpha=.55, rasterized=True)
                if len(predicted):
                    segments = np.stack([measured[mask, :2], predicted[:, :2]], axis=1)
                    ax.add_collection(LineCollection(segments, colors=error_colors[classes], linewidths=.45,
                                                     alpha=.20, rasterized=True))
                    for source, marker in [(0, 'o'), (1, '^')]:
                        take = record['source'][mask] == source
                        ax.scatter(predicted[take, 0], predicted[take, 1], s=11, marker=marker,
                                   c=error_colors[classes[take]], linewidths=0, alpha=.75, rasterized=True)
                ax.add_patch(Rectangle((-size[0] / 2, -size[1] / 2), size[0], size[1], fill=False,
                                       linestyle='--', linewidth=.7, edgecolor='#777777'))
                ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
                ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('Actor x (m)', fontsize=8)
                ax.set_ylabel('Actor y (m)', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.set_title(f'{label} | {entry["scene"]} / {owner[:8]}\n'
                             f'{entry["translation_speed_mps"]:.2f} m/s; build={len(build)}, owned={len(measured)}\n'
                             f'missing={int((~mask).sum())}; '
                             f'early/hit/late={int((classes == 0).sum())}/{int((classes == 1).sum())}/{int((classes == 2).sum())}',
                             fontsize=8.5, loc='left')
                if not len(measured):
                    ax.text(.5, .5, 'No owned heldout returns\nNo accuracy conclusion', transform=ax.transAxes,
                            ha='center', va='center', fontsize=9, bbox={'facecolor':'white', 'alpha':.85, 'edgecolor':'none'})
                ax.grid(alpha=.12)
                record.close()
        fig.suptitle(f'All 9 metadata-moving Actors: fixed first intersections ({page}/3)', y=.987, fontsize=13)
        fig.text(.5, .960, 'Canonical ground-plane projection; every stored owned beam retained; paired panels share limits',
                 ha='center', fontsize=9)
        fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .035), ncol=4, frameon=False, fontsize=8)
        fig.text(.5, .013, 'Segments connect endpoints of the same beam, not nearest neighbors. Dashed box is the annotation extent; height is not shown.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.065, right=.975, top=.91, bottom=.11, hspace=.46, wspace=.14)
        pdf.savefig(fig, dpi=170)
        fig.savefig(args.output / f'V73_MOVING_QUERY_INTERSECTIONS_{page}.png', dpi=150)
        plt.close(fig)
