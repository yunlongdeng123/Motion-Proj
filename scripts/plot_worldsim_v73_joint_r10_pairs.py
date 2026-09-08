"""直接绘制已保存的日志配对区间，不重新计算bootstrap；默认保留R10图。"""
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
parser.add_argument('--model-label', default='R10')
parser.add_argument('--reference', action='append', default=[], help='保存结果键=图中对照标签')
parser.add_argument('--protocol-note', default='R5 uses shorter labels and has a disclosed recovery RNG difference; R7/R11 share full-track labels and hard-free semantics.')
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
references = [item.split('=', 1) for item in args.reference] if args.reference else [
    ['joint_r5', 'R5 joint'], ['lidar_r7', 'R7 LiDAR'], ['native_r11', 'R11 native']]
refs = [item[0] for item in references]
positions = np.arange(len(refs))
for ax, (metric, title, scale) in zip(axes.flat, metrics):
    rows = [data['paired_final_minus'][ref]['development'][metric] for ref in refs]
    means = np.array([row['mean_delta'] for row in rows]) * scale
    intervals = np.array([row['bootstrap95'] for row in rows]) * scale
    ax.errorbar(means, positions, xerr=np.stack([means-intervals[:,0], intervals[:,1]-means]),
                fmt='o', color='#347f9e', capsize=4, markersize=5, linewidth=1.7)
    ax.axvline(0, color='#555555', ls=':', lw=1)
    ax.set(yticks=positions, yticklabels=['vs '+item[1] for item in references],
           title=title, ylim=(-.5,len(refs)-.5), xlabel=args.model_label+' minus reference')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax.title.set_fontsize(10)
    ax.tick_params(labelsize=9)
fig.suptitle(args.model_label+': paired development differences across five logs', fontsize=15, y=.98)
fig.text(.5,.927,'75 Actors retained; within-Actor observation weights, then independent-log means.',ha='center',fontsize=9)
fig.text(.5,.034,'Bars: saved 95% log bootstrap intervals (10,000 resamples, seed 7304). No new inference or resampling.',ha='center',fontsize=9)
fig.text(.5,.014,args.protocol_note,ha='center',fontsize=8.5)
fig.subplots_adjust(left=.11,right=.98,top=.86,bottom=.14,wspace=.48,hspace=.48)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output.with_suffix('.png'),dpi=180,facecolor='white')
fig.savefig(args.output.with_suffix('.pdf'),facecolor='white')
plt.close(fig)
