"""只绘保存的固定表面诊断；边际分类不冒充逐束状态迁移。"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(); parser.add_argument('--summary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
data=json.loads(args.summary.read_text())['statistics']
categories=[('first_hit','Correct first hit','#417d65'),
 ('early_with_later_hit','Early; later valid support','#e9b14d'),
 ('early_without_hit_but_near','Early; near surface only','#d98651'),
 ('early_without_hit_or_near','Early; no valid/near support','#b64e4e'),
 ('late_but_near','Late; near surface','#80b8cc'),
 ('late_without_near','Late; no near surface','#417eaa'),
 ('missing_but_near','Missing; near surface','#a6a4b9'),
 ('missing_without_near','Missing; no near surface','#d8d9df')]
fig,ax=plt.subplots(figsize=(12,4.3))
for y,(name,label) in enumerate([('lidar_r8','R8 LiDAR patches'),('mesh_lidar_r2','Q-v2 LiDAR mesh')]):
    left=0.
    for key,legend,color in categories:
        value=100*data[name]['equal_log_means'][key]
        ax.barh(y,value,left=left,height=.47,color=color,edgecolor='white',label=legend if y==0 else None)
        if value>=5: ax.text(left+value/2,y,f'{value:.1f}',ha='center',va='center',fontsize=9)
        left+=value
ax.set(yticks=[0,1],yticklabels=['R8 LiDAR patches','Q-v2 LiDAR mesh'],xlim=(0,100),
       xlabel='Share of original owned heldout rays (%)',ylim=(-.5,1.5))
ax.invert_yaxis(); ax.spines[['top','right','left']].set_visible(False); ax.grid(axis='x',alpha=.12)
fig.suptitle('Q-v2 LiDAR: more intersections do not imply correct first returns',fontsize=14,y=.98)
fig.text(.5,.89,'All 75 DEV Actors / 5 logs retained; Actor rates averaged within each log, then equal log weights.',ha='center',fontsize=9)
ax.legend(loc='upper center',bbox_to_anchor=(.42,-.29),ncol=4,fontsize=8.5,frameon=False)
fig.text(.5,.018,'Fixed stored meshes only. Near = unsigned surface distance <= 0.2 m; later hits never replace the first return.',ha='center',fontsize=9)
fig.subplots_adjust(left=.15,right=.98,top=.82,bottom=.33)
args.output.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+ext),dpi=180)
