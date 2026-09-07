from pathlib import Path
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'docs/autoresearch/worldsim_v73/m2/global')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
base=args.evidence; out=args.output
out.mkdir(parents=True,exist_ok=True)
runs={i:json.loads((base/f'r{i}_log_analysis.json').read_text()) for i in range(1,5)}
names=['LiDAR PCA','Native + LiDAR','R1: no depth anchor','R2: depth anchor','R3: beam tube','R4: extra fit times']
values=[runs[4]['stages'][key]['development'] for key in ['lidar_pca','native_lidar_fusion']]
values += [runs[i]['stages']['final']['development'] for i in range(1,5)]
colors=['#66788a','#96a7b8','#b96563','#386ca5','#bc8531','#418a71']
metrics=[('hit_rate','Literal hit rate (%) ↑',100),('early_rate','Owned-ray early rate (%) ↓',100),
         ('miss_rate','Owned-ray missing rate (%) ↓',100),('free_intrusion_m','All-ray free intrusion (m) ↓',1),
         ('surface_distance_m','Observed target-to-surface distance (m) ↓',1),('surface_recall_02','Observed coverage within 0.2 m (%) ↑',100)]
fig,axes=plt.subplots(2,3,figsize=(14.2,8.7))
for ax,(metric,title,factor) in zip(axes.flat,metrics):
    data=[v[metric]['mean']*factor for v in values]
    bars=ax.barh(names,data,color=colors,height=.7)
    ax.invert_yaxis(); ax.set_title(title,loc='left',fontsize=10.5,pad=10)
    ax.set_xlim(0,max(data)*1.22)
    ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    ax.spines[['right','top','left']].set_visible(False)
    ax.tick_params(axis='y',length=0,labelsize=8.8)
    for bar,val in zip(bars,data):
        label=f'{val:.1f}' if factor==100 else f'{val:.3f}'
        ax.text(val+max(data)*.018,bar.get_y()+bar.get_height()/2,label,va='center',fontsize=9)
fig.suptitle('V7.3 shared geometry: development trade-offs',x=.055,ha='left',fontsize=17)
fig.text(.055,.924,'Five existing development logs · one Actor per log · equal log means · full 24-view build context',fontsize=10,color='#44515c')
fig.text(.055,.032,'R1 loses native seed support; R2 restores direct depth supervision; R3 changes the free-space proxy; R4 adds fit-only sensor labels.\nR4 has a larger training-label budget. These are mechanism results, not fresh-log confirmation or full-population results. All methods use explicit triangle readout.',fontsize=9,color='#44515c')
fig.subplots_adjust(left=.16,right=.97,top=.86,bottom=.13,wspace=.68,hspace=.35)
for ext in ['png','pdf']: fig.savefig(out/f'V73_GLOBAL_MECHANISM_TRADEOFFS.{ext}',dpi=180,facecolor='white')
print(out/'V73_GLOBAL_MECHANISM_TRADEOFFS.png')
