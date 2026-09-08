"""Separate fixed-background changes from the Actor model comparison."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--pca',type=Path,required=True)
parser.add_argument('--pca-joint',type=Path,required=True)
parser.add_argument('--vdb',type=Path,required=True)
parser.add_argument('--vdb-carved',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
def read(path): return json.loads(path.read_text())['statistics']
pca=read(args.pca); pca['joint_r5']=read(args.pca_joint)['joint_r5']
vdb=read(args.vdb); carved=read(args.vdb_carved)
fig,axes=plt.subplots(2,3,figsize=(13,7.7))
bg_colors=['#5190a9','#bca177','#be7752']
for ax,metric,title,factor in zip(axes[0],['hit_rate','miss_rate','free_intrusion_m'],
    ['Background only: all-ray hit','Background only: all-ray missing','Background only: all-ray free intrusion'],[100,100,1]):
    maximum=0
    for i,(source,color) in enumerate(zip([pca,vdb,carved],bg_colors)):
        value=source['background_only']['all_raw_returns']
        mean=value['means'][metric]*factor
        logs=np.array([v[metric] for v in value['per_log'].values()])*factor
        ax.bar(i,mean,width=.60,color=color,alpha=.75)
        ax.scatter(i+np.linspace(-.1,.1,len(logs)),logs,color=color,s=22,edgecolor='white',linewidth=.5,zorder=3)
        ax.text(i,mean,f'{mean:.1f}' if factor==100 else f'{mean:.3f}',ha='center',va='bottom',fontsize=8)
        maximum=max(maximum,max(logs))
    ax.set_xticks(range(3),['PCA +\nbuild carve','TSDF','TSDF +\nbuild carve'],fontsize=8)
    ax.set_ylim(0,maximum*1.18); ax.set_title(title,loc='left',fontsize=10)
    ax.set_ylabel('%' if factor==100 else 'm')
methods=['lidar_pca','native_fusion','r9','joint_r5']
for ax,metric,title,factor in zip(axes[1],['hit_rate','early_rate','free_intrusion_m'],
    ['Actor-owned rays: hit','Actor-owned rays: early return','Actor-owned rays: free intrusion'],[100,100,1]):
    maximum=0
    for j,(source,color,label) in enumerate([(pca,bg_colors[0],'PCA + build carve'),(carved,bg_colors[2],'TSDF + build carve')]):
        for i,method in enumerate(methods):
            value=source[method]['cohort_returns']; mean=value['means'][metric]*factor
            logs=np.array([v[metric] for v in value['per_log'].values()])*factor
            x=i+(j-.5)*.36
            ax.bar(x,mean,width=.32,color=color,alpha=.75,label=label if i==0 else None)
            ax.scatter(x+np.linspace(-.055,.055,len(logs)),logs,color=color,s=15,edgecolor='white',linewidth=.4,zorder=3)
            maximum=max(maximum,max(logs))
    ax.set_xticks(range(4),['LiDAR\nPCA','Native\nfusion','LiDAR\nr9','Joint\nr5'],fontsize=8)
    ax.set_ylim(0,maximum*1.18); ax.set_title(title,loc='left',fontsize=10)
    ax.set_ylabel('%' if factor==100 else 'm')
for ax in axes.flat:
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
fig.suptitle('TSDF background reduces false occlusion, with a coverage cost',y=.985,fontsize=14)
fig.text(.5,.943,'6 development scenes / 5 logs; 416,704 original beams; 11,886 Actor-owned beams. Bars = equal log means, dots = logs.',ha='center',fontsize=9)
handles,labels=axes[1,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,frameon=False,fontsize=9)
fig.text(.5,.022,'All Actor surfaces, original evaluation beams and trajectories are fixed. Background construction reads build scans only; no learned opacity.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.065,right=.975,top=.88,bottom=.15,hspace=.50,wspace=.27)
args.output.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+ext),dpi=170)
