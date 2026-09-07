"""真实场景首交点与背景近点修正的静态研究图，不把框代理当真实实例边界。"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser(); parser.add_argument('--before',type=Path,required=True)
parser.add_argument('--after',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
before=json.loads(args.before.read_text())['statistics']; after=json.loads(args.after.read_text())['statistics']
methods=['background_only','lidar_pca','lidar_r6','native_fusion_r2']; labels=['Background\nonly','LiDAR\nPCA','LiDAR query\nr6','Native depth\nfusion r2']
fig,axes=plt.subplots(2,3,figsize=(12,7.5))
for ax,(group,metric,label,scale) in zip(axes.flat,[
    ('cohort_returns','hit_rate','Cohort return: literal hit (%)',100),
    ('cohort_returns','early_rate','Cohort return: early (%)',100),
    ('cohort_returns','miss_rate','Cohort return: missing (%)',100),
    ('background_proxy_returns','early_rate','Background proxy: early (%)',100),
    ('background_proxy_returns','free_intrusion_m','Background proxy: intrusion (m)',1),
    ('all_raw_returns','miss_rate','All raw returns: missing (%)',100)]):
    x=np.arange(len(methods)); width=.36
    for values,offset,color,name in [(before,-width/2,'#afbbc7','Initial background'),(after,width/2,'#236c75','Build sensor-close points excluded')]:
        y=[values[m][group]['means'][metric]*scale for m in methods]
        ax.bar(x+offset,y,width,color=color,label=name)
    ax.set_xticks(x,labels,fontsize=8); ax.set_title(label,fontsize=10)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
handles,legend_labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,legend_labels,fontsize=9,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
fig.suptitle('Scene composition: the fixed background is itself a source of false early hits',fontsize=14,y=.98)
fig.text(.05,.035,'6 existing development scenes / 5 logs. Ray-weighted within scene, then equal scene/log means; point estimates.\n'
    'All 416,704 evaluation rays retained. Only static build background changes; Actor surfaces and trajectories stay fixed.\n'
    'Box ownership and background are proxies. LiDAR query and native fusion have different fitting-label budgets.',fontsize=9)
fig.tight_layout(rect=(0,.11,1,.90))
fig.savefig(args.output/'V73_SCENE_BACKGROUND_COMPOSITION.png',dpi=180)
fig.savefig(args.output/'V73_SCENE_BACKGROUND_COMPOSITION.pdf')
