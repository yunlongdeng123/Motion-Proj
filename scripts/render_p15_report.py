"""P1.5 可导出的架构、连续诊断和真实表面图。"""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_worldsim_v74_h2_p15 import load,surface,write
from motion_proj.worldsim_v74_h2.first_event_trace import trace

out=Path(sys.argv[1]);figdir=ROOT/'docs/figures/worldsim_v74_h2_p15';figdir.mkdir(parents=True,exist_ok=True)
source=Path(json.loads((out/'manifest.json').read_text())['source']);fit=source/'20260912__fit-teacher-r3'/'fit'
rows=json.loads((out/'transitions.json').read_text());stable=json.loads((out/'stable_hit_transitions.json').read_text());ips=json.loads((out/'interpolations.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors={'A':'#c75042','C2':'#287ca5','teacher':'#4a8a67'}
def save(fig,name):
    fig.savefig(figdir/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(12.6,2.7));ax.set(xlim=(-.025,1.025),ylim=(0,1));ax.axis('off')
blocks=[(.005,.44,.19,.43,'Saved states + rays\nA / C2: steps 0–8\n12 exposed FIT tasks'),(.25,.44,.205,.43,'Hard triangle query\nHIT → EARLY + owner\n3 geometric margins'),(.515,.44,.20,.43,'Matched-parent teacher\nSO(3) / center / radius\ninterpolation'),(.78,.44,.215,.43,'Failure attribution\nowner / depth / tie\nNo new training')]
for x,y,w,h,label in blocks:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',facecolor='#edf3f6',edgecolor='#476678'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
for i in range(3):
    a,b=blocks[i],blocks[i+1];ax.annotate('',xy=(b[0]-.008,.655),xytext=(a[0]+a[2]+.008,.655),arrowprops={'arrowstyle':'->','lw':1.8,'color':'#476678'})
ax.text(.5,.19,'Finding in thin cases: unchanged first-hit owner + normal translation → amplified range error',ha='center',va='center',color='#9d3d33',fontsize=12)
save(fig,'architecture')

fig,axes=plt.subplots(1,3,figsize=(12,3.45))
fr=[r for r in rows if r['model']=='A' and r['family']=='grazing_thin'];sr=[r for r in stable if r['model']=='A' and r['family']=='grazing_thin']
specs=[('First–second hit gap (mm)',lambda r:r['first_hit_margin_m']*1000,10),('Distance to outer boundary (mm)',lambda r:r['old_before']['boundary_m']*1000,10),('Incidence |n · d|',lambda r:r['old_before']['incidence'],.1)]
for ax,(label,fn,cut) in zip(axes,specs):
    for subset,color,text in [(sr,'#9aa9b1','HIT → HIT (1,400)'),(fr,colors['A'],'First HIT → EARLY (52)')]:
        x=np.sort([fn(r) for r in subset]);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,label=text,lw=2)
    ax.axvline(cut,color='#3e5667',ls=':',label='Fixed near-boundary scale');ax.set(xlabel=label,ylabel='Empirical CDF',ylim=(0,1.03));ax.grid(alpha=.15)
axes[0].legend(fontsize=8,loc='lower right');fig.suptitle('A / grazing_thin: failure is not concentrated at support or first-hit switching boundaries',fontsize=12)
fig.tight_layout();save(fig,'margins')

# 覆盖最早退化、后续退化和另一失败对象；各组选择中心排序的射线，选择规则可复现。
selections=[]
for name,step in [('grazing_thin-00',4),('grazing_thin-00',6),('grazing_thin-01',8)]:
    group=sorted([r for r in fr if r['case']==name and r['step']==step],key=lambda r:r['after_m']-r['observed_m'])
    selections.append(group[len(group)//2])
write(out/'figure_selection.json',dict(rule='Predeclared after audit: earliest thin failure, later same-case failure, other failed thin case; median post-transition residual within each group',cases=selections))
fig,axes=plt.subplots(2,3,figsize=(13,7.2))
for col,r in enumerate(selections):
    name,step,ray=r['case'],r['step'],r['ray'];obs=load(fit/name/'supervision.npz');y=obs['observed_first_range_m'][ray]
    ax=axes[0,col]
    for model in ['A','C2']:
        folder=source/f'20260912__{model}-dagger12-r1'/name
        q=np.array([load(folder/f'query_chain-{i}.npz')['first'][ray] for i in range(9)])-y
        ax.plot(np.arange(9),q,'o-',color=colors[model],label=model,lw=2,ms=4)
    ax.axhspan(-.2,.2,color='#dcecdf',alpha=.65);ax.axhline(-.2,color='#465e54',ls=':');ax.axvline(step,color='#777',ls=':',lw=1)
    ax.set(title=f'{name}, ray {ray}, step {step}',xlabel='Saved closed-loop step',ylabel='First-hit residual (m)',xticks=range(9));ax.grid(alpha=.15);ax.legend(fontsize=9)
    ax=axes[1,col];dest=out/'A'/name/f'step-{step}';z=load(dest/'teacher_learner_scan.npz');q=z['first_m'][:,ray]-y
    ax.plot(z['alpha'],q,color=colors['A'],lw=2,label='First-hit range')
    ax.axhspan(-.2,.2,color='#dcecdf',alpha=.65);ax.axhline(-.2,color='#465e54',ls=':')
    ip=next(x for x in ips if x['model']=='A' and x['case']==name and x['step']==step and x['ray']==ray)
    ae=ip['early']['alpha_hi'];ax.axvline(ae,color='#c75042',ls=':');ax.text(.04,.08,f'EARLY begins at alpha={ae:.3f}\nNo owner switch at EARLY',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    changed=np.flatnonzero(z['winner'][1:,ray]!=z['winner'][:-1,ray])+1
    for i in changed:ax.plot(z['alpha'][i],q[i],marker='x',color='#287ca5',ms=8,label='Owner changes (still HIT)' if i==changed[0] else None)
    ax.set(xlabel='Teacher → learner alpha',ylabel='First-hit residual (m)',xlim=(0,1));ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper right')
fig.suptitle('Continuous range drift: selected saved trajectories and matched-teacher interpolations',fontsize=13);fig.tight_layout();save(fig,'trajectories_and_interpolation')

fig=plt.figure(figsize=(12.8,10.3))
for row,r in enumerate([selections[0],selections[-1]]):
    name,step,ray=r['case'],r['step'],r['ray'];dest=out/'A'/name/f'step-{step}';obs=load(fit/name/'supervision.npz');o=obs['origins_actor_m'][ray];d=obs['directions_actor'][ray];y=obs['observed_first_range_m'][ray]
    states=[surface(dest/'parent.npz'),surface(dest/'learner.npz'),surface(dest/'teacher.npz')]
    vs=np.concatenate([s.mesh()[0] for s in states]);lo=vs.min(0)-.08;hi=vs.max(0)+.08
    for col,(s,title) in enumerate(zip(states,['Before learner update','After learner update','Teacher from same parent'])):
        ax=fig.add_subplot(2,3,row*3+col+1,projection='3d',computed_zorder=False);v,f=s.mesh();polys=Poly3DCollection(v[f],alpha=.3,linewidth=.4,edgecolor='#667986',zorder=1);polys.set_facecolor(['#8dc0d4' if k//8==r['winner_before'] else '#d6d1c8' for k in range(len(f))]);ax.add_collection3d(polys)
        tr=trace(s,obs);q=tr['first'][ray];line=o+np.array([y-.45,y+.35])[:,None]*d;ax.plot(*line.T,color='#263b4a',lw=1.7,zorder=3)
        ax.scatter(*(o+y*d),s=35,color='#347a52',marker='o',label='Measured return',zorder=4,depthshade=False);ax.scatter(*(o+q*d),s=45,color=colors['A'],marker='x',label='Actual first hit',zorder=5,depthshade=False)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=f'{title}\n{name}, residual={q-y:+.3f}m')
        ax.view_init(elev=18,azim=-55);ax.set_box_aspect(hi-lo);ax.tick_params(labelsize=7)
        if row==0 and col==0:ax.legend(fontsize=7,loc='upper left')
fig.suptitle('Saved physical surfaces: center motion changes depth without changing the winning patch',fontsize=13,y=.97);fig.subplots_adjust(hspace=.4,wspace=.15,top=.9,bottom=.08);save(fig,'physical_surfaces')
print(str(figdir))
