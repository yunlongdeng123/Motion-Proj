"""Fixed-output point/surface density sensitivity, independent log means and dots."""
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
colors=['#8b6eac','#458fb5','#d09a37','#b55553']
def surface_logs(stage,metric):
    values=defaultdict(list)
    for row in data['actors'][stage]:
        if row[metric] is not None: values[row['log_id']].append(row[metric])
    return [float(np.mean(v)) for v in values.values()]
def point_logs(name,metric):
    return [v[metric] for v in data['point_statistics'][name]['per_log'].values() if v[metric] is not None]
def panel(ax,values,labels,palette,title,unit,factor=1):
    x=np.arange(len(values)); means=[np.mean(v)*factor for v in values]
    ax.bar(x,means,color=palette,alpha=.72,width=.60)
    for i,v in enumerate(values):
        ax.scatter(i+np.linspace(-.13,.13,len(v)),np.array(v)*factor,s=23,color=palette[i],edgecolors='white',linewidths=.65,zorder=3)
        ax.text(i,means[i],f'{means[i]:.1f}' if factor==100 else f'{means[i]:.3f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(x,labels,fontsize=8); ax.set_title(title,loc='left',fontsize=11,pad=10)
    ax.set_ylabel(unit,fontsize=9); ax.grid(axis='y',alpha=.17); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False); ax.set_ylim(0,max(max(v)*factor for v in values)*1.19)
fig,axes=plt.subplots(2,3,figsize=(12,7.2))
labels=['Matched\ncenters','Native\npoints','Matched\nsurface','Full-density\nsurface']
panel(axes[0,0],[point_logs('matched_centers','distance_m'),point_logs('native_points','distance_m'),surface_logs('matched','surface_distance_m'),surface_logs('dense','surface_distance_m')],labels,colors,'Measured target-to-geometry distance','m')
panel(axes[0,1],[point_logs('matched_centers','recall_02'),point_logs('native_points','recall_02'),surface_logs('matched','surface_recall_02'),surface_logs('dense','surface_recall_02')],labels,colors,'Measured coverage within 0.2 m','%',100)
for ax,metric,title,factor in [(axes[0,2],'hit_rate','Literal hit',100),(axes[1,0],'early_rate','Early return',100),
                              (axes[1,1],'miss_rate','Missing return',100),(axes[1,2],'free_intrusion_m','Observed free-space intrusion',1)]:
    panel(ax,[surface_logs('matched',metric),surface_logs('dense',metric)],['Matched surface','Full-density surface'],colors[2:],title,'%' if factor==100 else 'm',factor)
fig.suptitle('AdaPoinTr Y-up r2: fixed output density changes coverage and physical errors',fontsize=14,y=.98)
fig.text(.5,.933,'75 development Actors / 5 independent logs; bars = equal log means, dots = individual logs',ha='center',fontsize=10)
fig.text(.5,.039,'Matched: min(build points, 1024) + 512 patches. Full density: 16,384 patches per nonempty Actor. Same 0.06 m PCA patch scale.',ha='center',fontsize=9)
fig.text(.5,.014,'Fixed saved outputs, no retraining. Density sensitivity is separate from matched-budget comparisons; sparse targets do not define complete surface precision.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.07,right=.98,top=.865,bottom=.14,hspace=.48,wspace=.32)
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+extension),dpi=170)
