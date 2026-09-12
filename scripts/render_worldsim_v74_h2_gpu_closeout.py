from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
R=Path('/root/autodl-tmp/motion_proj');B=Path('/root/autodl-tmp/runs/worldsim_v74_h2/WS-V74-H2-GPU-P1-01');O=R/'docs/figures/worldsim_v74_h2_gpu';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(15,4));ax.set_xlim(0,15);ax.set_ylim(0,4);ax.axis('off')
boxes=[(.15,1.5,2.1,1.15,'BUILD witnesses\npositive endpoints\nnegative ray prefixes'),(2.8,1.5,2.35,1.15,'Shared surface state\nposition + SO(3)\n8 support radii'),(5.75,1.5,2.4,1.15,'GPU full competition\nA: ordered messages\nC2: same-info set'),(8.75,1.5,2.3,1.15,'Geometry update\ncontinuous parameters\nbirth/split, 8 steps'),(11.65,1.5,3.1,1.15,'Frozen finite triangles\nliteral nearest intersection\nreusable across QUERY rays')]
for x,y,w,h,label in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',fc='#e9f3f4',ec='#306a73',lw=1.5));ax.text(x+w/2,y+h/2,label,ha='center',va='center')
for i in range(4):
    x,y,w,h,_=boxes[i];nx=boxes[i+1][0];ax.add_patch(FancyArrowPatch((x+w+.05,y+h/2),(nx-.07,y+h/2),arrowstyle='-|>',mutation_scale=16,color='#306a73',lw=1.5))
ax.annotate('retrace after every geometric update',xy=(3.95,1.43),xytext=(9.7,.55),ha='center',arrowprops={'arrowstyle':'->','connectionstyle':'angle,angleA=0,angleB=-90,rad=6','color':'#306a73'},color='#306a73')
ax.text(7.5,3.45,'FIT-only hard event teacher: BUILD + supervision; DAgger labels learner-visited surfaces',ha='center',color='#825936')
ax.annotate('',xy=(7,2.75),xytext=(7,3.2),arrowprops={'arrowstyle':'->','linestyle':'--','color':'#825936'})
ax.text(7.5,.05,'Current implementation stops at P1 learnability; no real-data or method-novelty claim',ha='center',color='#8b3535')
fig.savefig(O/'architecture.png',dpi=160,bbox_inches='tight');fig.savefig(O/'architecture.pdf',bbox_inches='tight');plt.close(fig)

data={name:json.loads((B/f'20260912__{name}-dagger12-r1/results.json').read_text()) for name in ['A','C2']}
teacher=json.loads((B/'20260912__fit-teacher-r3/teacher_results.json').read_text());names={r['case'] for r in data['A']};data['FIT teacher']=[r for r in teacher if r['case'] in names]
families=['multilayer','shared_support','missing_support','grazing_thin'];colors=['#d17142','#308594','#678d60']
fig,axes=plt.subplots(1,3,figsize=(14,4.4))
for ax,metric in zip(axes,['hit','early','miss']):
    x=np.arange(4)
    for j,(name,rows) in enumerate(data.items()):
        vals=[np.mean([r['query'][metric] for r in rows if r['family']==f])*100 for f in families]
        ax.bar(x+(j-1)*.23,vals,.22,label=name,color=colors[j])
    ax.set_xticks(x,labels=['layers','shared','missing','thin']);ax.set_title('QUERY '+metric);ax.set_ylim(0,105 if metric=='hit' else 30);ax.set_ylabel('% of positive returns');ax.spines[['top','right']].set_visible(False)
axes[-1].legend(frameon=False);fig.suptitle('Twelve memorized synthetic FIT tasks after the one bounded correction\nNot an independent test; A retains its pre-correction best checkpoint',fontsize=13)
fig.tight_layout();fig.savefig(O/'learnability.png',dpi=160);fig.savefig(O/'learnability.pdf');plt.close(fig)

ordered=sorted(data['A'],key=lambda r:r['query']['hit']-r['initial_query']['hit']);selected=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
fig=plt.figure(figsize=(11,10))
for row,record in enumerate(selected):
    name=record['case'];paths=[B/'20260912__fit-teacher-r3/fit'/name/'initial.npz',B/'20260912__A-dagger12-r1'/name/'step-8.npz',B/'20260912__C2-dagger12-r1'/name/'step-8.npz']
    assets=[]
    for path in paths:
        with np.load(path) as z:assets.append((z['vertices_actor_m'],z['faces']))
    allv=np.concatenate([v for v,f in assets]);center=(allv.max(0)+allv.min(0))/2;extent=max(np.ptp(allv,axis=0).max()/2,.1)*1.1
    for col,((v,f),label) in enumerate(zip(assets,['INITIAL','A','C2'])):
        ax=fig.add_subplot(3,3,row*3+col+1,projection='3d');ax.add_collection3d(Poly3DCollection(v[f],facecolor=['#9da7b0','#d17142','#308594'][col],edgecolor='#56636a',linewidth=.2))
        for axis,k in zip(['x','y','z'],range(3)):getattr(ax,'set_'+axis+'lim')(center[k]-extent,center[k]+extent)
        ax.set_box_aspect((1,1,1));ax.view_init(25,-65);ax.set_axis_off();ax.set_title(label)
        if col==0:ax.text2D(0,-.02,name,transform=ax.transAxes,fontsize=9)
fig.suptitle('Worst / median / best A hit-gain among the same twelve FIT tasks\nOne view and bounds per case; opaque fixed assets',fontsize=13);fig.tight_layout();fig.savefig(O/'failure_surfaces.png',dpi=150);plt.close(fig)
(O/'case_selection.json').write_text(json.dumps({'rule':'sort A QUERY hit gain vs initial, select worst/middle/best; 12 FIT tasks only','cases':[r['case'] for r in selected]},indent=2)+'\n')
print(O)
