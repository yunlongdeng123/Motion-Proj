"""直接绘制R10已保存的日志配对区间，不重新计算bootstrap。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--analysis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = json.loads(args.analysis.read_text(encoding='utf-8'))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.7))
metrics = [('hit_rate', 'Hit rate (pp; higher better)', 100),
           ('early_rate', 'Early rate (pp; lower better)', 100),
           ('miss_rate', 'Missing rate (pp; lower better)', 100),
           ('free_intrusion_m', 'Free-space intrusion (m; lower better)', 1),
           ('surface_distance_m', 'Target-to-surface distance (m; lower better)', 1),
           ('surface_recall_02', 'Surface recall (pp; higher better)', 100)]
refs = ['joint_r5', 'lidar_r7', 'native_r11']
for ax, (metric, title, scale) in zip(axes.flat, metrics):
    rows = [data['paired_final_minus'][ref]['development'][metric] for ref in refs]
    means = np.array([row['mean_delta'] for row in rows]) * scale
    intervals = np.array([row['bootstrap95'] for row in rows]) * scale
    ax.errorbar(means, np.arange(3), xerr=np.stack([means-intervals[:,0], intervals[:,1]-means]),
                fmt='o', color='#347f9e', capsize=4, markersize=5, linewidth=1.7)
    ax.axvline(0, color='#555555', ls=':', lw=1)
    ax.set(yticks=np.arange(3), yticklabels=['vs R5 joint', 'vs R7 LiDAR', 'vs R11 native'],
           title=title, ylim=(-.5,2.5), xlabel='R10 minus reference')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax.title.set_fontsize(10)
    ax.tick_params(labelsize=9)
fig.suptitle('R10: paired development differences across five logs', fontsize=15, y=.98)
fig.text(.5,.927,'75 Actors retained; within-Actor observation weights, then independent-log means.',ha='center',fontsize=9)
fig.text(.5,.034,'Bars: saved 95% log bootstrap intervals (10,000 resamples, seed 7304). No new inference or resampling.',ha='center',fontsize=9)
fig.text(.5,.014,'R5 uses shorter labels and has a disclosed recovery RNG difference; R7/R11 share full-track labels and hard-free semantics.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.11,right=.98,top=.86,bottom=.14,wspace=.48,hspace=.48)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output.with_suffix('.png'),dpi=180,facecolor='white')
fig.savefig(args.output.with_suffix('.pdf'),facecolor='white')
plt.close(fig)
