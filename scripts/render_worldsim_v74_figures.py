"""固定评价结果的科研图与失败对象视图，不用于调参。"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
parser=argparse.ArgumentParser();parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--quality-only',action='store_true');parser.add_argument('--skip-cases',action='store_true');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
data=json.loads(args.summary.read_text());plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
colors={'WEX':'#1269a3','RIF':'#e47723','DCS':'#33956c','A3_milp':'#6a95b5','B2_regular':'#d8a477','C1_residual':'#8bb8a1','V73_R8':'#75519b','NKSR_ks_native':'#a34457'}
offsets={'WEX':(5,-13),'A3_milp':(5,7),'V73_R8':(5,-14),'C1_residual':(5,7),'DCS':(5,-12),'B2_regular':(-70,7),'RIF':(5,6),'NKSR_ks_native':(5,6)}
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for i,ds in enumerate(['nuscenes','av2']):
    for method,stats in data['summary'][ds].items():
        if method not in colors:continue
        q=stats['ready']['metrics_query'];hit=q['hit_rate']['mean'];free=q['mean_free_intrusion_m']['mean'];recall=q['positive_surface_recall_02']['mean'];early=q['early_rate']['mean']
        if None in [hit,free,recall]:continue
        marker='o' if method in ['WEX','RIF','DCS'] else '^'
        for ax,x,label in [(axes[i,0],early*100,'Early-hit rate (%), lower is better'),(axes[i,1],free,'Free-space intrusion (m), lower is better'),(axes[i,2],recall*100,'Recall @ 0.2m (%), higher is better')]:
            ax.scatter(x,hit*100,c=colors[method],marker=marker,s=70,label=method)
            if method in ['WEX','RIF','DCS','NKSR_ks_native']:ax.annotate(method,(x,hit*100),xytext=offsets[method],textcoords='offset points',fontsize=8)
            ax.set_xlabel(label);ax.set_ylabel('Held-out first-hit rate (%)');ax.grid(alpha=.2);ax.set_title(ds+' / own FIT, exposed DEV');ax.margins(x=.20,y=.15)
for ax in axes[:,:2].flat:ax.set_xlim(left=0)
fig.suptitle('V74: coverage and physical consistency must improve together\nLog-equal means; native external/historical face budgets are reported separately',fontsize=13)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.055),ncol=4,fontsize=9)
fig.savefig(args.output/'quality_tradeoffs.png',bbox_inches='tight');fig.savefig(args.output/'quality_tradeoffs.pdf',bbox_inches='tight');plt.close(fig)
if args.quality_only:raise SystemExit(0)

fig,ax=plt.subplots(figsize=(12,4.2));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=color,edgecolor='#304052',linewidth=1.2));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#52606d',linewidth=1.3))
box(.1,1.2,1.7,1.0,'Object BUILD\npoints + rays','#e9eef3')
box(2.65,2.6,2.4,.65,'A: shared domain\nresponsibility exchange','#d8e9f5');box(2.65,1.5,2.4,.65,'B: interval FEM field','#fce7d5');box(2.65,.4,2.4,.65,'C: demand-driven\npatch proposals','#dcefe3')
for y in [2.925,1.825,.725]:arrow((1.9,1.7),(2.5,y));arrow((5.2,y),(6.1,1.8))
box(6.25,1.25,1.8,1.1,'Fixed opaque\ntriangle surface','#e9eef3');box(8.9,1.25,2.4,1.1,'True first intersection\nheld-out evaluation','#e9eef3');arrow((8.2,1.8),(8.75,1.8))
ax.text(3.85,3.7,'Independent candidates + matched controls',ha='center',fontsize=12);ax.text(9.95,.55,'QUERY truth stays in evaluator',ha='center',fontsize=10)
fig.savefig(args.output/'architecture_components.png',bbox_inches='tight');fig.savefig(args.output/'architecture_components.pdf',bbox_inches='tight');plt.close(fig)
if args.skip_cases:raise SystemExit(0)

rows=[];cases={}
for source in data['sources']:
    folder=Path(source);rows+=json.loads((folder/'per_actor.json').read_text())
    for case in json.loads((folder/'manifest.json').read_text())['cases']:cases[(case['dataset'],case['case_id'])]=case
selected=[]
for method,control in [('WEX','A3_milp'),('RIF','B2_regular'),('DCS','C1_residual')]:
    candidates=[]
    for row in rows:
        if row['dataset']!='nuscenes' or row['method']!=method or row.get('contract_missing_input') or row.get('metrics_query') is None:continue
        ref=next((r for r in rows if r['dataset']==row['dataset'] and r['case_id']==row['case_id'] and r['method']==control),None)
        if ref is None or ref.get('metrics_query') is None:continue
        hit=row['metrics_query']['hit_rate'];rhit=ref['metrics_query']['hit_rate']
        if hit is not None and rhit is not None:candidates.append((hit-rhit,row,ref))
    if not candidates:continue
    delta,row,ref=sorted(candidates,key=lambda t:(t[0],t[1]['case_id']))[0];case=cases[(row['dataset'],row['case_id'])]
    with np.load(case['build_file']) as z:build=z['support_points_actor_m']
    with np.load(case['query_truth_file']) as z:query=z['points_actor_m'][z['positive_actor'].astype(bool)]
    meshes=[]
    for r in [row,ref]:
        with np.load(r['surface_path']) as z:meshes.append((z['vertices_actor_m'],z['faces']))
    points=np.vstack([build,query]+[v[np.unique(f)] for v,f in meshes if len(f)]);lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;radius=max((hi-lo).max()/2,.25)*1.05
    fig=plt.figure(figsize=(11,5.5));fig.suptitle(f'{method} vs {control}: lowest paired nuScenes hit difference ({delta*100:+.2f} pp)\n{row["case_id"]}',fontsize=10)
    fig.subplots_adjust(left=.04,right=.93,bottom=.19,top=.80,wspace=.15)
    for i,r in enumerate([row,ref]):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        v,f=meshes[i]
        if len(f):ax.add_collection3d(Poly3DCollection(v[f],facecolor='#6494b0',edgecolor='#39576b',linewidths=.1,alpha=.18))
        ax.scatter(*build.T,s=9,c='#222222',label='BUILD',depthshade=False);ax.scatter(*query.T,s=13,c='#d44935',label='held-out returns',depthshade=False)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-62)
        q=r['metrics_query'];ax.set_title(f'{r["method"]}: hit {q["hit_rate"]*100:.1f}%, early {q["early_rate"]*100:.1f}%');ax.tick_params(labelsize=8)
    fig.legend(*ax.get_legend_handles_labels(),ncol=2,loc='lower center',bbox_to_anchor=(.5,.055),fontsize=8)
    fig.text(.5,.015,'Diagnostic views use identical axes/camera. Display alpha is for visibility; geometric readout is fully opaque.',ha='center',fontsize=8)
    fig.savefig(args.output/(method+'_failure_case.png'),bbox_inches='tight');plt.close(fig)
    selected.append({'method':method,'control':control,'selection_rule':'minimum paired nuScenes owned-query hit delta, ties case_id; diagnostics only','case_id':row['case_id'],'delta':delta,'candidate_surface':row['surface_path'],'control_surface':ref['surface_path'],'build_file':case['build_file'],'query_truth_file':case['query_truth_file']})
(args.output/'figure_sources.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
print('Research figures saved',args.output)
