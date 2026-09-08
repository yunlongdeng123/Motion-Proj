"""Joint population geometry and literal-readout tradeoff, with independent logs."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(); parser.add_argument('--summary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
data=json.loads(args.summary.read_text())
methods=['initial','final','lidar_r6','native_fusion','lidar_event_r9']
labels=['Joint\ninitial','Joint\nr5','LiDAR\nr6','Native\nfusion','LiDAR\nr9']
colors=['#bfb2a4','#bb655c','#498ca4','#b38c41','#6f9977']
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for ax,(metric,title,factor) in zip(axes.flat,[('hit_rate','Literal hit',100),('early_rate','Early return',100),
    ('miss_rate','Missing return',100),('free_intrusion_m','Observed free-space intrusion',1),
    ('surface_distance_m','Measured target-to-surface distance',1),('surface_recall_02','Measured surface recall within 0.2 m',100)]):
    values=[]
    for method in methods:
        logs=defaultdict(list)
        for row in data['actors'][method]:
            if row['role']=='development' and row[metric] is not None: logs[row['log_id']].append(row[metric])
        values.append(np.array([np.mean(v) for v in logs.values()])*factor)
    means=[np.mean(v) for v in values]; x=np.arange(len(methods))
    ax.bar(x,means,color=colors,width=.63,alpha=.75)
    for i,value in enumerate(values):
        ax.scatter(i+np.linspace(-.12,.12,len(value)),value,color=colors[i],edgecolor='white',linewidth=.6,s=23,zorder=3)
        ax.text(i,means[i],f'{means[i]:.1f}' if factor==100 else f'{means[i]:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,labels,fontsize=7.5); ax.set_ylabel('%' if factor==100 else 'm')
    ax.set_title(title,loc='left',fontsize=10); ax.grid(axis='y',alpha=.16); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False); ax.set_ylim(0,max(max(v) for v in values)*1.17)
fig.suptitle('Joint native adaptation + spatial queries: coverage gains with physical errors',fontsize=14,y=.98)
fig.text(.5,.936,'75 development Actors / 5 independent logs, including empty inputs; bars = equal log means, dots = individual logs',ha='center',fontsize=9.5)
fig.text(.5,.039,'r5 versus r6 shares short-window labels. Native fusion uses build-only head adaptation; r9 uses full-track labels and different physical objectives.',ha='center',fontsize=8.5)
fig.text(.5,.015,'Fixed patch budget and literal surface readout. Target-to-surface distance is one-sided; coverage improvement does not establish correct first intersections.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.065,right=.98,top=.865,bottom=.15,wspace=.29,hspace=.50)
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+extension),dpi=170)
