"""只绘已保存诊断结果，不重复几何查询或统计重采样。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = Path(__file__).resolve().parents[1] / 'docs/autoresearch/worldsim_v73/qv2'
joint = json.loads((p / 'support_r4/summary.json').read_text())['statistics']['mesh_joint_r1']['equal_log_means']
lidar = json.loads((p / 'support_r3/summary.json').read_text())['statistics']['mesh_lidar_r2']['equal_log_means']
mesh = json.loads((p / 'mesh_diagnostic_r1/summary.json').read_text())['statistics']
categories = ['first_hit', 'early_with_later_hit', 'early_without_hit_but_near', 'early_without_hit_or_near',
              'late_but_near', 'late_without_near', 'missing_but_near', 'missing_without_near']
labels = ['Correct first hit', 'Early; later correct support', 'Early; near target, no ray hit',
          'Early; no correct or near support', 'Late; near target', 'Late; no near target',
          'Missing; near target', 'Missing; no near target']
colors = ['#3a9479', '#ddb86b', '#e58c59', '#bb4f43', '#7fa8c7', '#436d94', '#c2b8d4', '#7b668f']
fig = plt.figure(figsize=(14, 8.8))
grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=.63, wspace=.38)
ax = fig.add_subplot(grid[0, :])
left = np.zeros(2)
for key, label, color in zip(categories, labels, colors):
    values = np.array([joint[key], lidar[key]]) * 100
    ax.barh([1, 0], values, left=left, color=color, label=label, height=.55)
    for y, width, start in zip([1, 0], values, left):
        if width >= 4:
            ax.text(start + width/2, y, f'{width:.1f}', ha='center', va='center', color='white', fontsize=10)
    left += values
ax.set(yticks=[1, 0], yticklabels=['Q-v2 joint r1', 'Q-v2 LiDAR r2'], xlim=(0, 100),
       xlabel='Log-equal fractions of owned heldout returns (%)', title='Most joint early rays have no later correct support')
ax.legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, -.30), fontsize=9, frameon=False)
ax2 = fig.add_subplot(grid[1, 0])
keys = ['smin_p05', 'smax_p95', 'anisotropy_p95']
for offset, (name, label, color) in zip([-.18, .18], [('mesh_joint_r1', 'Joint r1', '#397d9e'), ('mesh_lidar_r2', 'LiDAR r2', '#d19648')]):
    vals = [mesh[name]['equal_log_means_of_actor_statistics'][k] for k in keys]
    bars = ax2.bar(np.arange(3) + offset, vals, width=.34, label=label, color=color)
    ax2.bar_label(bars, fmt='%.2f', padding=3)
ax2.set(xticks=np.arange(3), xticklabels=['s_min p05', 's_max p95', 'Anisotropy p95'],
        ylabel='Mean Actor quantile, log-equal', ylim=(0, 4.8), title='Local deformation relative to analytic template')
ax2.legend(frameon=False)
ax3 = fig.add_subplot(grid[1, 1])
counts = [mesh[k]['actors_with_nonadjacent_intersection'] for k in ['mesh_joint_r1', 'mesh_lidar_r2']]
bars = ax3.bar(['Joint r1', 'LiDAR r2'], counts, color=['#397d9e', '#d19648'], width=.55)
ax3.bar_label(bars, labels=[f'{n} / 67' for n in counts], padding=5)
ax3.set(ylim=(0, 67), ylabel='Nonempty Actors (raw count)', title='Detected nonadjacent triangle intersections')
for axis in [ax, ax2, ax3]:
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('Q-v2 fixed-mesh diagnosis: support location remains the joint bottleneck', y=.98, fontsize=16)
fig.text(.5, .017, '75 DEV Actors / 5 logs; 8 empty meshes retain missing predictions but have no distortion statistic.\n'
         'Template is not ground truth. Intersection test skips pairs sharing any vertex; zero detections do not certify no folding.\n'
         'Saved meshes only. No neural inference, mesh edits, new data, or repeated LiDAR ray queries.', ha='center', fontsize=10)
fig.subplots_adjust(top=.90, bottom=.14, left=.12, right=.98)
for ext in ['png', 'pdf']:
    fig.savefig(p / ('V73_QV2_FIXED_DIAGNOSTICS.' + ext), dpi=160)
print('saved fixed diagnostic figure')
