"""Publication figure from saved scene metrics; no repeated model evaluation."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--before',type=Path,required=True)
parser.add_argument('--after',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
old=json.loads(args.before.read_text())['statistics']; new=json.loads(args.after.read_text())['statistics']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,3,figsize=(12,7.2))
colors=['#8194a6','#167b88']
metrics=[('free_intrusion_m','Free-space intrusion (m)',1),('early_rate','Early return (%)',100),('miss_rate','Missing return (%)',100)]
for ax,(metric,label,factor) in zip(axes[0],metrics):
    before=old['background_only']['all_raw_returns']; after=new['background_only']['all_raw_returns']
    logs=list(before['per_log'])
    for log in logs:
        values=[before['per_log'][log][metric]*factor,after['per_log'][log][metric]*factor]
        ax.plot([0,1],values,'o-',color='#a3aab0',alpha=.7,lw=1,ms=4)
    ax.plot([0,1],[before['means'][metric]*factor,after['means'][metric]*factor],
            'o-',color=colors[1],lw=2.5,ms=7,label='Five-log mean')
    ax.set_xticks([0,1],['Original background','Build-free carved'])
    ax.set_ylabel(label); ax.set_title('Background only · all observed beams')
    ax.grid(axis='y',alpha=.18); ax.set_xlim(-.2,1.2); ax.set_ylim(bottom=0)
axes[0,0].legend(frameon=False,fontsize=9)
methods=['lidar_pca','r8','r9','capa_r2','adapointr_r1','native_fusion']
labels=['LiDAR\nPCA','r8','r9\nevent','CAPA\nr2','Ada\nr1*','Native\nfusion']
for ax,(metric,label,factor) in zip(axes[1],[metrics[0],('hit_rate','Hit return (%)',100),metrics[2]]):
    for side,stages,color in zip([-.18,.18],[old,new],colors):
        values=[stages[method]['cohort_returns']['means'][metric]*factor for method in methods]
        ax.bar(np.arange(len(methods))+side,values,width=.34,color=color,
               label='Original' if side<0 else 'Carved',alpha=.9)
    # Individual logs for the carved geometry, preserving independent sample count.
    for i,method in enumerate(methods):
        values=[v[metric]*factor for v in new[method]['cohort_returns']['per_log'].values()]
        ax.scatter(i+.18+np.linspace(-.08,.08,len(values)),values,s=10,color='#202c33',alpha=.7,zorder=3)
    ax.set_xticks(np.arange(len(methods)),labels); ax.set_ylabel(label)
    ax.set_title('Scene composition\nActor-owned return proxy')
    ax.grid(axis='y',alpha=.18); ax.set_ylim(bottom=0)
axes[1,0].legend(frameon=False,fontsize=9)
fig.suptitle('Build-only background carving reduces intrusion with a coverage cost',fontsize=14,y=.98)
fig.text(.055,.017,'6 scenes / 5 development logs. Bars and thick lines: log means; dots: individual logs. All observed beams retained.\n'
         '* Ada r1 retains the PCN axis limitation. Methods have different training/adaptation budgets; no final joint-model result shown.',fontsize=9)
fig.tight_layout(rect=[0,.065,1,.95],h_pad=2.0,w_pad=1.6)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(str(args.output)+'.png',dpi=180)
fig.savefig(str(args.output)+'.pdf')
plt.close(fig)
