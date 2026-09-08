"""按真实训练模式绘制已存过程证据；冻结梯度不画成可学习曲线。"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(); parser.add_argument('--summary',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--model-label'); args=parser.parse_args()
data=json.loads(args.summary.read_text()); rows=data['epochs']; epochs=[r['epoch'] for r in rows]
fig,axes=plt.subplots(2,3,figsize=(12,7.2))
blue='#347f9e'; orange='#c18035'
def line(ax,values,label,color=blue): ax.plot(epochs,values,label=label,color=color,lw=1.8)
line(axes[0,0],[r['native_huber_measurement_weighted_m'] for r in rows],'Measurement weighted')
line(axes[0,0],[r['native_huber_supervised_actor_distribution_m']['mean'] for r in rows],'Supervised Actor mean',orange)
axes[0,0].set_title('Native build-depth Huber'); axes[0,0].set_ylabel('m'); axes[0,0].legend(fontsize=8)
if all(r['native_huber_measurement_weighted_m'] is None for r in rows):
    axes[0,0].text(.5,.5,'No native build-depth supervision',ha='center',va='center',
                   transform=axes[0,0].transAxes,fontsize=9)
for ax,metric,title in [(axes[0,1],'coverage_m','Sampled target-to-surface distance'),(axes[0,2],'free_intrusion_m','Sampled hard free-space intrusion')]:
    line(ax,[r['metrics'][metric]['mean'] for r in rows],'Actor mean')
    line(ax,[r['metrics'][metric]['median'] for r in rows],'Actor median',orange)
    ax.set_title(title); ax.set_ylabel('m'); ax.legend(fontsize=8)
if data.get('training_mode')=='lidar_only':
    axes[1,0].text(.5,.5,'Not applicable: LiDAR-only inputs\nNo native support selection',ha='center',va='center',
                   transform=axes[1,0].transAxes,fontsize=9)
else:
    line(axes[1,0],[100*r['predicted_support_fallback_with_views']/r['actors_with_views'] if r['actors_with_views'] else 0 for r in rows],'No in-box native support')
axes[1,0].set_title('LiDAR fallback among Actors with views'); axes[1,0].set_ylabel('% of presentations')
zero_groups=[]; plotted_gradients=False
for group,label,color in [('native_dpt','Native DPT',blue),('query_decoder','Query decoder',orange)]:
    values=[r['gradient_groups'][group]['median'] for r in rows]
    if any(value is not None and value>0 for value in values):
        line(axes[1,1],[value if value is not None and value>0 else None for value in values],label,color)
        plotted_gradients=True
    else:
        zero_groups.append(label)
axes[1,1].set_title('Median parameter-gradient group norm'); axes[1,1].set_ylabel('Before global clipping (log scale)')
if plotted_gradients:
    axes[1,1].set_yscale('log'); axes[1,1].legend(fontsize=8)
for i,label in enumerate(zero_groups):
    axes[1,1].text(.5,.08+i*.08,label+': no positive median gradient',ha='center',
                   transform=axes[1,1].transAxes,fontsize=8,
                   bbox={'facecolor':'white','edgecolor':'none','alpha':.92,'pad':2})
line(axes[1,2],[r['metrics']['step_s']['mean'] for r in rows],'Mean step time')
axes[1,2].set_title('Observed step time under shared resources'); axes[1,2].set_ylabel('s / Actor presentation')
for ax in axes.flat:
    ax.set_xlabel('Epoch'); ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
    ax.title.set_fontsize(10)
    if 'resume' in data: ax.axvline(data['resume']['completed_epochs']+.5,color='#777777',ls=':',lw=1)
mode=data.get('training_mode')
title={'native_only':'Native geometry decoder + canonical fusion',
       'joint':'Joint native geometry + spatial queries',
       'pointwise':'Native geometry + pointwise queries',
       'lidar_only':'LiDAR spatial queries'}.get(mode,'Shared geometry model')
fig.suptitle((args.model_label or title)+': training process',fontsize=14,y=.98)
counts=[r['actor_presentations'] for r in rows]
count_text=str(counts[0]) if len(set(counts))==1 else str(min(counts))+'--'+str(max(counts))
fig.text(.5,.936,count_text+' fit presentations per recorded epoch; sampled observations vary, not paired generalization evidence.',ha='center',fontsize=9)
if 'resume' in data:
    footer='Dotted line: recovery after epoch '+str(data['resume']['completed_epochs'])+'. Completed epochs are counted once.'
    discarded=data.get('parent_interruption',{}).get('unsaved_presentations')
    if discarded is not None: footer+=' '+str(discarded)+' discarded parent updates remain recorded.'
else:
    footer='No checkpoint resume. Actual optimizer updates: '+str(data['optimizer_updates_in_saved_history'])+'.'
    footer+=' Skipped presentations: '+str(data['total_presentations_in_saved_history']-data['optimizer_updates_in_saved_history'])+'.'
fig.text(.5,.035,footer,ha='center',fontsize=8.5)
fig.text(.5,.014,'Native loss excludes unobserved Actors. Memory contention and concurrent work affect timing; loss reduction does not establish physical accuracy.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.075,right=.98,top=.87,bottom=.13,wspace=.34,hspace=.43)
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ['png','pdf']: fig.savefig(args.output.with_suffix('.'+extension),dpi=170)
