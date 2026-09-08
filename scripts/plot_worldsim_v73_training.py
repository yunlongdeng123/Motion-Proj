"""Training-process evidence from saved epochs; no heldout or inferred success."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(); parser.add_argument('--summary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
data=json.loads(args.summary.read_text()); rows=data['epochs']; epochs=[r['epoch'] for r in rows]
fig,axes=plt.subplots(2,3,figsize=(12,7.2))
blue='#347f9e'; orange='#c18035'
def line(ax,values,label,color=blue): ax.plot(epochs,values,label=label,color=color,lw=1.8)
line(axes[0,0],[r['native_huber_measurement_weighted_m'] for r in rows],'Measurement weighted')
line(axes[0,0],[r['native_huber_supervised_actor_distribution_m']['mean'] for r in rows],'Supervised Actor mean',orange)
axes[0,0].set_title('Native build-depth Huber'); axes[0,0].set_ylabel('m'); axes[0,0].legend(fontsize=8)
for ax,metric,title in [(axes[0,1],'coverage_m','Sampled target-to-surface distance'),(axes[0,2],'free_intrusion_m','Sampled hard free-space intrusion')]:
    line(ax,[r['metrics'][metric]['mean'] for r in rows],'Actor mean')
    line(ax,[r['metrics'][metric]['median'] for r in rows],'Actor median',orange)
    ax.set_title(title); ax.set_ylabel('m'); ax.legend(fontsize=8)
line(axes[1,0],[100*r['predicted_support_fallback_with_views']/r['actors_with_views'] if r['actors_with_views'] else 0 for r in rows],'No in-box native support')
axes[1,0].set_title('LiDAR fallback among Actors with views'); axes[1,0].set_ylabel('% of presentations')
for group,label,color in [('native_dpt','Native DPT',blue),('query_decoder','Query decoder',orange)]:
    line(axes[1,1],[r['gradient_groups'][group]['median'] for r in rows],label,color)
axes[1,1].set_title('Median parameter-gradient group norm'); axes[1,1].set_ylabel('Before global clipping (log scale)')
axes[1,1].set_yscale('log'); axes[1,1].legend(fontsize=8)
line(axes[1,2],[r['metrics']['step_s']['mean'] for r in rows],'Mean step time')
axes[1,2].set_title('Observed step time under shared resources'); axes[1,2].set_ylabel('s / Actor presentation')
for ax in axes.flat:
    ax.set_xlabel('Epoch'); ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
    ax.title.set_fontsize(10)
    if 'resume' in data: ax.axvline(data['resume']['completed_epochs']+.5,color='#777777',ls=':',lw=1)
fig.suptitle('Joint native geometry + spatial queries: training process',fontsize=14,y=.98)
fig.text(.5,.936,'371 fit Actors per complete epoch; sampled targets and rays vary, so this is not paired generalization evidence.',ha='center',fontsize=9)
fig.text(.5,.035,'Dotted line: recovery from epoch 21. Saved complete epochs are counted once; 107 discarded parent updates remain separately recorded.',ha='center',fontsize=8.5)
fig.text(.5,.014,'Native loss excludes unobserved Actors. Memory contention and concurrent work affect timing; loss reduction does not establish physical accuracy.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.075,right=.98,top=.87,bottom=.13,wspace=.34,hspace=.43)
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+extension),dpi=170)
