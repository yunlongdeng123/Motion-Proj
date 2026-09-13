"""P1.6 科学图；所有图从已保存结果生成。"""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from audit_worldsim_v74_h2_p16 import load,surface,describe,trace
p=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
read=lambda f:json.loads((p/f).read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white','pdf.fonttype':42})
def save(fig,name):
    fig.savefig(dest/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(dest/f'{name}.pdf',bbox_inches='tight',pad_inches=.15);plt.close(fig)
fig,ax=plt.subplots(figsize=(14,5));ax.set_xlim(-.2,14.2);ax.set_ylim(-1.25,4.2);ax.axis('off')
def box(x,y,w,h,text,color='#e9eff8'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',fc=color,ec='#516377',lw=1.2));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.5,color='#516377'))
box(0,1.3,2.4,1.4,'Saved A / C2 states\nBUILD observations\nFrozen checkpoints')
box(3.3,2.3,3.1,1.1,'Normal / tangent error\nRay-conditioned depth\nDrift and recovery')
box(3.3,.15,3.1,1.2,'Frozen model\nBUILD + current surface\nPredict next update')
box(7.3,.15,2.7,1.2,'Normal least squares\nBUILD first-hit metric\nFixed 8-step solver')
box(10.8,1.2,3,1.5,'Hard FIT ray evaluation\nHit + miss + free space\nResiduals and boundaries','#e3f1e9')
arrow((2.55,2.25),(3.15,2.75));arrow((2.55,1.6),(3.15,.85));arrow((6.55,.75),(7.15,.75));arrow((10.15,.75),(10.7,1.65));arrow((6.55,2.8),(10.65,2.35))
ax.annotate('',xy=(4.85,.01),xytext=(8.65,.01),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.2',lw=1.3,color='#516377'))
ax.text(6.75,-.68,'Corrected surface feeds next model step',ha='center',fontsize=10,color='#516377')
ax.text(7.3,3.9,'12 exposed FIT tasks; no training',ha='center',fontsize=13,weight='bold')
ax.text(7.3,-1.07,'BUILD+FIT solver is a separately labelled diagnostic, never a deployable control.',ha='center',fontsize=10,color='#536273')
save(fig,'architecture')

teacher=read('teacher_error_rows.json');trans=read('ray_transitions.json');fail={(r['model'],r['case'],r['step'],r['ray']) for r in trans if r['category']=='hit_early'}
fig,axes=plt.subplots(1,2,figsize=(11.6,4.3),layout='constrained')
for ax,m in zip(axes,['A','C2']):
    rr=[r for r in teacher if r['model']==m and r['family']=='grazing_thin' and (m,r['case'],r['step'],r['ray']) in fail]
    for k,label,color in [('normal_m','Normal center error','#197f75'),('tangent_m','Tangent center error','#d89e2c'),('normal_depth_m','Normal / incidence','#be3b43')]:
        x=np.sort([abs(r['decomposition'][k])*1000 for r in rr]);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label,color=color,lw=2)
    ax.axvline(200,ls='--',c='gray',lw=1);ax.text(195,.45,'200 mm HIT tolerance',rotation=90,ha='right',fontsize=9,color='gray')
    ax.set(xlim=(0,285),ylim=(0,1.05),title=f'{m}: teacher-to-learner at first EARLY ({len(rr)} rays)',xlabel='Absolute error (mm)',ylabel='Cumulative fraction of rays');ax.grid(alpha=.2);ax.legend(loc='center',bbox_to_anchor=(.46,.3),fontsize=8.5)
fig.suptitle('Small mixed-unit loss can conceal a >200 mm ray error\nSame parent and existing patch; exposed thin-structure tasks',fontsize=13)
save(fig,'physical_errors')

fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
selection=[('A','grazing_thin-01',5),('C2','grazing_thin-00',8),('A','grazing_thin-00',66)]
# 第一幅在 thin-01 失败射线中选第一条，第三幅为新引入非正例侵入的第一条。
selection[0]=('A','grazing_thin-01',min(r['ray'] for r in trans if r['model']=='A' and r['case']=='grazing_thin-01' and r['category']=='hit_early'))
for ax,(m,n,ray) in zip(axes,selection):
    obs=load(p/'inputs'/n/'supervision.npz');y=obs['observed_first_range_m'][ray]
    for mode,label,color in [('raw','Original frozen rollout','#bb3947'),('closed_build','With BUILD metric control','#237c9b')]:
        q=np.array([load(p/mode/m/n/f'query_chain-{t}.npz')['first'][ray] for t in range(9)]);q[~np.isfinite(q)]=np.nan
        ax.plot(range(9),q-y,'o-',color=color,lw=2,ms=4,label=label)
    if ray==8 and m=='C2':
        a=surface(p/'raw'/m/n/'step-6.npz');ti=describe(a,obs,trace(a,obs));k=int(ti['winner'][ray]);plane=[]
        for t in range(6,9):
            st=surface(p/'raw'/m/n/f'step-{t}.npz');normal=st.rotation[k,:,2];plane.append(normal@(st.center[k]-obs['origins_actor_m'][ray])/(normal@obs['directions_actor'][ray])-y)
        ax.plot(range(6,9),plane,'x--',c='#dc9821',lw=1.5,label='Old patch plane after step 6')
    if bool(obs['positive_actor'][ray]):ax.axhspan(-.2,.2,color='#dff0e1',alpha=.6)
    ax.axhline(-.2,color='gray',ls=':',lw=1);ax.set(title=f'{m} / {n}\nray {ray}'+(' (non-positive)' if not obs['positive_actor'][ray] else ''),xlabel='Surface update step',ylabel='First range minus observation (m)',xticks=range(9));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    if ray==66:ax.text(.5,.15,'Original: no surface hit\nControl: new free-space intrusion',transform=ax.transAxes,ha='center',fontsize=9)
fig.suptitle('Drift correction, support-based recovery, and a new negative-ray failure',fontsize=14)
save(fig,'trajectories');(p/'figure_selection.json').write_text(json.dumps(selection,indent=2)+'\n')

summary=read('control_summary.json');fig,axes=plt.subplots(2,4,figsize=(14.2,7),layout='constrained')
spec=[('hit','HIT (%)',100),('miss','MISS (%)',100),('recall_02','Geometry recall (%)',100),('free_m','Mean free intrusion (mm)',1000)]
for i,m in enumerate(['A','C2']):
    for j,(key,title,scale) in enumerate(spec):
        ax=axes[i,j];raw=summary['raw'][m]['all']['final'][key]*scale;ctrl=summary['closed_build'][m]['all']['final'][key]*scale
        bars=ax.bar(['Original','Metric control'],[raw,ctrl],color=['#aab4c5','#237c9b'],width=.58)
        for b,v in zip(bars,[raw,ctrl]):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom',fontsize=10)
        ax.set(title=f'{m}: {title}',ylim=(0,max(raw,ctrl)*1.18 if max(raw,ctrl)>0 else 1));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Restoring the 70 original failure rays does not establish joint success\nMacro means over the same 12 exposed FIT tasks',fontsize=14)
save(fig,'joint_metrics')
print('Rendered architecture and three scientific figures')
