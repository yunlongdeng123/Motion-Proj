"""Plot saved Actor TSDF coverage and literal-return outcomes; no inference."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--construction', type=Path, required=True)
parser.add_argument('--analysis', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
construction = json.loads(args.construction.read_text())['construction']
analysis = json.loads(args.analysis.read_text())
args.output_dir.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1, 1.35]})
bins = [(0, 1), (1, 16), (16, 64), (64, 256), (256, float('inf'))]
groups = [[row for row in construction if low <= row['actor']['build_points'] < high] for low, high in bins]
positive = [sum(row['carving']['retained_triangles'] > 0 for row in rows) for rows in groups]
fraction = [n / len(rows) for n, rows in zip(positive, groups)]
pca = {row['owner']: row for row in analysis['actors']['lidar_pca']}
pca_available = [sum(pca[row['actor']['owner']]['surface_patches'] > 0 for row in rows) / len(rows) for rows in groups]
ax = axes[0]
ax.bar(np.arange(5), np.asarray(fraction)*100, color='#2b7092', width=.65, label='TSDF native mesh')
ax.plot(np.arange(5), np.asarray(pca_available)*100, 'o--', color='#858585', ms=4, label='LiDAR PCA availability')
for x, n, rows, f in zip(range(5), positive, groups, fraction):
    ax.text(x, f*100+3, f'{n}/{len(rows)}', ha='center', fontsize=9)
ax.set_xticks(range(5), ['0', '1–15', '16–63', '64–255', '≥256'])
ax.set_ylim(0, 114)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Actors with a nonempty surface (%)')
ax.set_xlabel('Original build LiDAR points per Actor')
ax.set_title('(a) Build support: all 489 Actors / 25 logs', loc='left', fontsize=10)
ax.legend(loc='upper left', bbox_to_anchor=(0, .79), frameon=True, facecolor='white', edgecolor='white', framealpha=.95, fontsize=8)

ax = axes[1]
names = ['lidar_pca', 'final', 'native_lidar_fusion', 'lidar_r9', 'joint_r5']
labels = ['LiDAR\nPCA', 'TSDF\n+ carve', 'Native\nfusion', 'LiDAR\nR9', 'Joint\nR5']
stats = [analysis['stages'][name]['development'] for name in names]
hit = np.array([s['hit_rate']['mean'] for s in stats])
early = np.array([s['early_rate']['mean'] for s in stats])
miss = np.array([s['miss_rate']['mean'] for s in stats])
late = np.maximum(1-hit-early-miss, 0)
bottom = np.zeros(5)
for values, label, color in [(hit, 'Hit', '#358469'), (early, 'Early', '#bc514b'),
                             (late, 'Late', '#d5ab56'), (miss, 'Missing', '#d6dde1')]:
    ax.bar(range(5), values*100, bottom=bottom*100, width=.68, color=color, label=label)
    bottom += values
for x, value in enumerate(miss):
    ax.text(x, 1.02*100, f'{value*100:.1f}% miss', ha='center', fontsize=8)
ax.set_xticks(range(5), labels)
ax.set_ylim(0, 116)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel('Literal first-return outcome (%)')
ax.set_title('(b) Development: 75 Actors / 5 logs', loc='left', fontsize=10)
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=4, frameon=False, fontsize=9)
fig.subplots_adjust(left=.065, right=.985, top=.90, bottom=.25, wspace=.28)
fig.text(.065, .065, 'Right: Actor-level ray counts → log means → equal log weight. Owned misses retained; 23 Actors have no owned heldout return.', fontsize=8)
fig.text(.065, .025, 'No hole filling: low TSDF intrusion accompanies sparse output support. Native triangle density is not matched to learned query patches.', fontsize=8)
for extension in ['png', 'pdf']:
    fig.savefig(args.output_dir / ('V73_ACTOR_TSDF.' + extension), dpi=160)
plt.close(fig)
