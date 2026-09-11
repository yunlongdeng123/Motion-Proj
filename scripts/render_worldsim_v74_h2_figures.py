import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
def box(ax,x,y,w,h,label,color='#eaf2fa'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',facecolor=color,edgecolor='#3c5265',linewidth=1.2))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
def arrow(ax,p,q,label=None,dashed=False):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,linewidth=1.3,color='#3c5265',linestyle='--' if dashed else '-'))
    if label:ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.12,label,ha='center',fontsize=8)
def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(14,5));ax.set(xlim=(0,14),ylim=(0,5));ax.axis('off')
    ax.text(.2,4.65,'V74-H2  |  Shared surface dynamics',fontsize=17,weight='bold')
    ax.text(.2,4.27,'Candidate architecture; full event policy and multi-step evidence are still pending.',fontsize=10,color='#536273')
    box(ax,.2,2.5,2,1,'BUILD rays\norigins, directions,\nmeasured returns')
    box(ax,2.85,2.5,2,1,'Shared finite surface\ncenters + rotations\n8 boundary radii')
    box(ax,5.5,2.5,2,1,'Exact event tracing\nall intersections\n+ boundary neighbors')
    box(ax,8.2,2.5,2.2,1,'Ray-to-patch update\nA: ordered messages\nC2: complete set')
    box(ax,11.15,2.5,2.5,1,'One frozen mesh\n8 actual triangles / patch\n4096-face budget','#eaf5eb')
    for x,y in [(2.2,2.85),(4.85,5.5),(7.5,8.2),(10.4,11.15)]:arrow(ax,(x,3),(y,3))
    arrow(ax,(9.3,2.5),(9.3,1.9));arrow(ax,(9.3,1.9),(3.85,1.9),'joint geometry update / re-trace');arrow(ax,(3.85,1.9),(3.85,2.5))
    box(ax,.2,.45,3,0.8,'FIT-only held-out frames\nteacher / training loss','#fff4dd')
    arrow(ax,(3.2,.85),(9.3,.85),'supervision only',True);arrow(ax,(9.3,.85),(9.3,1.9),dashed=True)
    box(ax,11.15,.45,2.5,.8,'New ray + rigid pose\nnearest geometric hit','#eaf5eb');arrow(ax,(12.4,2.5),(12.4,1.25))
    fig.tight_layout();fig.savefig(a.output/'architecture.png',dpi=170);fig.savefig(a.output/'architecture.pdf');plt.close(fig)
    rows=json.loads((a.cpu/'summary.json').read_text())['rows'];families=['multilayer','shared_support','missing_support','grazing_thin'];methods=['INITIAL','C1','C3','TRUTH_ASSISTED','C6_TINY']
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True);colors=['#8d98a6','#007a91','#bf6d1d','#417947','#bc5161']
    for ax,key,title in zip(axes,['hit','early','miss'],['Correct first hit (%)','Early first hit (%)','No intersection (%)']):
        for j,(method,color) in enumerate(zip(methods,colors)):
            values=[next(r for r in rows if r['family']==f and r['method']==method)['query'][key]*100 for f in families]
            ax.barh(np.arange(4)+(j-2)*.14,values,height=.13,color=color,label=method)
        ax.set_xlim(0,105);ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_yticks(range(4),['Multi-layer','Shared support','Missing support','Grazing / thin'])
    fig.suptitle('CPU reference mechanisms: 20 scenes per family; learned A/C2 not evaluated here',fontsize=12)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,frameon=False)
    fig.tight_layout(rect=[0,.1,1,.93]);fig.savefig(a.output/'cpu_controls.png',dpi=170);fig.savefig(a.output/'cpu_controls.pdf');plt.close(fig)
    # 预定三个展示：C1相对初始化命中改善的最差、中位、最好，不能只选漂亮个例。
    full=json.loads((a.cpu/'results.json').read_text());by={r['case']:{x['method']:x for x in full if x['case']==r['case']} for r in full}
    ordered=sorted(by,key=lambda c:(by[c]['C1']['query']['hit']-by[c]['INITIAL']['query']['hit'],c));chosen=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,8));selection=[]
    for i,c in enumerate(chosen):
        meshes=[]
        for name in ['INITIAL','C1','TRUTH_ASSISTED']:
            with np.load(a.cpu/c/(name+'.npz')) as z:meshes.append((z['vertices_actor_m'],z['faces']))
        allv=np.concatenate([v for v,f in meshes]);lo=allv.min(0);hi=allv.max(0);center=(lo+hi)/2;span=max(float(np.max(hi-lo)),1.)*.58
        for j,(name,(v,f)) in enumerate(zip(['INITIAL','C1','TRUTH_ASSISTED'],meshes)):
            ax=fig.add_subplot(3,3,i*3+j+1,projection='3d');ax.add_collection3d(Poly3DCollection(v[f],facecolor=['#9ca3ad','#238d9d','#54a05d'][j],edgecolor='#4c5960',linewidth=.15,alpha=1.))
            ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span);ax.set_box_aspect([1,1,1]);ax.view_init(20,-65);ax.set_axis_off()
            ax.set_title(f'{name}\nhit={by[c][name]["query"]["hit"]:.1%}',fontsize=10)
            if j==0:ax.text2D(0,.0,c,transform=ax.transAxes,fontsize=8)
        selection.append({'case':c,'criterion':'C1 minus INITIAL query hit: minimum / median / maximum'})
    fig.suptitle('Fixed shared meshes: preselected worst / median / best C1 gain',fontsize=13);fig.tight_layout();fig.savefig(a.output/'case_surfaces.png',dpi=160);plt.close(fig)
    (a.output/'case_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
if __name__=='__main__':main()
