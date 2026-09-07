"""完整开发Actor队列的表面/物理取舍；保存已有逐日志结果，不重跑实验。"""
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'docs/autoresearch/worldsim_v73/m2/global/population_lidar_r8_analysis.json').read_text())
methods=['lidar_pca','lidar_r6','lidar_r7','final']
names=['LiDAR PCA','R6 short labels','R7 full-track','R8 metric beam']
colors=['#788390','#46a8aa','#3564b2','#cb713c']
panels=[('hit_rate','Literal hit (%) ↑',100),('early_rate','Early returns (%) ↓',100),
        ('miss_rate','Missing returns (%) ↓',100),('free_intrusion_m','Free-space intrusion (cm) ↓',100),
        ('surface_distance_m','Measured target to surface (cm) ↓',100),('surface_recall_02','Surface recall @ 20 cm (%) ↑',100)]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(12,7.4))
for ax,(metric,title,factor) in zip(axes.flat,panels):
    mean_labels=[]
    for i,(method,color) in enumerate(zip(methods,colors)):
        grouped=defaultdict(list)
        for row in data['actors'][method]:
            if row['role']=='development' and row[metric] is not None: grouped[row['log_id']].append(row[metric]*factor)
        values=[float(np.mean(grouped[log])) for log in sorted(grouped)]
        mean=data['stages'][method]['development'][metric]['mean']*factor
        ax.bar(i,mean,width=.64,color=color,alpha=.72)
        ax.scatter(np.array([i]*len(values))+np.linspace(-.14,.14,len(values)),values,
                   s=18,facecolor='white',edgecolor=color,linewidth=1,zorder=3)
        mean_labels.append(f'{["PCA","R6","R7","R8"][i]}\n{mean:.2f}')
    ax.set_title(title,loc='left',fontsize=11)
    ax.set_xticks(range(4),mean_labels)
    ax.set_ylim(bottom=0); ax.margins(y=.18); ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
fig.suptitle('V7.3: lower intrusion with a coverage tradeoff',fontsize=16,y=.985)
fig.legend([plt.Rectangle((0,0),1,1,color=c,alpha=.72) for c in colors],names,
           loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False)
fig.text(.055,.047,'75 development Actors / 5 existing logs. Bars: equal-log means; circles: individual logs. Empty predictions remain included.',fontsize=9)
fig.text(.055,.019,'R6 uses short-window fit targets; R7/R8 use full-track fit targets. R8 changes only the free-space objective relative to R7.',fontsize=9)
fig.subplots_adjust(top=.845,bottom=.16,hspace=.46,wspace=.26)
folder=ROOT/'docs/autoresearch/worldsim_v73/m2/global'
for extension in ['png','pdf']: fig.savefig(folder/('V73_POPULATION_FREE_TRADEOFFS.'+extension),dpi=180)
