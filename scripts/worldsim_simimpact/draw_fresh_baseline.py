"""全部八起点的实测基线；上下文、组件图和连续轨迹误差同时报告。"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
e=a.evidence;base=json.loads((e/'fit_admission.json').read_text())['rows'][0];route=json.loads((e/'fit_admission_route.json').read_text())['rows'][0]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
colors={'gt':'#2b7461','fixed':'#b94b3c','route':'#3979b6'}
fig=plt.figure(figsize=(18,13),facecolor='white');gs=fig.add_gridspec(4,4,height_ratios=[.56,1.5,1.15,1.15],left=.055,right=.98,top=.94,bottom=.105,hspace=.55,wspace=.33)
fig.suptitle('Real-sensor baseline must qualify before testing reconstruction harm',x=.055,ha='left',fontsize=22,fontweight='bold')
ax=fig.add_subplot(gs[0,:]);ax.axis('off')
boxes=[(.02,'Real RGB + LiDAR\nSame causal ego state'),(.265,'Fixed straight /\nGT-derived route hint'),(.51,'Frozen TransFuser\nOfficial PDM tracking'),(.755,'Compare with log\n4 s / all 8 starts')]
for x,txt in boxes:
    ax.add_patch(FancyBboxPatch((x,.17),.215,.66,boxstyle='round,pad=.012',fc='#eef3f7',ec='#aac0cc',transform=ax.transAxes));ax.text(x+.1075,.5,txt,ha='center',va='center',transform=ax.transAxes,fontsize=12)
for x,_ in boxes[:-1]:ax.annotate('',xy=(x+.242,.5),xytext=(x+.22,.5),arrowprops={'arrowstyle':'->','lw':1.7},xycoords='axes fraction')
ax=fig.add_subplot(gs[1,:2]);ax.imshow(Image.open(e/'scene0002_start00_front.jpg'));ax.axis('off');ax.set_title('(a) scene-0002: original front camera, first registered start',loc='left',fontsize=13,fontweight='bold')
ax.text(0,-.06,'Metadata-selected interaction; this image is context, not a phantom-surface claim.',transform=ax.transAxes,fontsize=11)
ax=fig.add_subplot(gs[1,2:]);x=np.arange(8)
for row,c,label in [(base,colors['fixed'],'Fixed straight'),(route,colors['route'],'Route hint (extra GT bit)')]:
    values=[r['FDE_vs_recorded_ego_m'] for r in row['individual_outcomes']];ax.plot(x,values,'o-',color=c,label=label,lw=2)
ax.axhline(2,ls='--',c='#89939a',label='Mean FDE admission: 2 m');ax.set_xticks(x,[f'{i:02d}' for i in x]);ax.set_xlabel('Registered start');ax.set_ylabel('4 s final displacement error (m)');ax.set_title('(b) Residual baseline error remains',loc='left',fontsize=13,fontweight='bold');ax.legend(fontsize=10,loc='upper left');ax.set_ylim(0,8)
ax.text(.99,.98,f"Mean ADE / FDE\nFixed: {base['mean_ADE_m']:.2f} / {base['mean_FDE_m']:.2f} m\nRoute: {route['mean_ADE_m']:.2f} / {route['mean_FDE_m']:.2f} m\nOverlap starts: 1/8 → 0/8",transform=ax.transAxes,va='top',ha='right',fontsize=11,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
for j in range(8):
    ax=fig.add_subplot(gs[2+j//4,j%4]);case=f'scene-0002_start{j:02d}'
    b=np.load(e/'baseline'/case/'states.npz');r=np.load(e/'baseline_route_r1'/case/'states.npz');gt=b['ground_truth_ego'];bb=b['simulated'][0];rr=r['simulated'][0]
    assert np.allclose(gt,r['ground_truth_ego'])
    for v,c,label in [(gt,colors['gt'],'Recorded ego'),(bb,colors['fixed'],'Fixed straight'),(rr,colors['route'],'Route hint')]:
        ax.plot(v[:,0],v[:,1],color=c,label=label,lw=2);ax.scatter(v[-1,0],v[-1,1],c=c,s=22,zorder=4)
    ax.scatter([0],[0],c='#263743',marker='^',s=35,zorder=6);ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('Forward (m)');ax.set_ylabel('Left (m)');ax.grid(alpha=.18)
    ax.set_title(f"Start {j:02d} · FDE {base['individual_outcomes'][j]['FDE_vs_recorded_ego_m']:.2f} → {route['individual_outcomes'][j]['FDE_vs_recorded_ego_m']:.2f} m",fontsize=11,loc='left')
    if base['individual_outcomes'][j]['actor_overlap_any']:ax.text(.03,.92,'Fixed-command box overlap',c=colors['fixed'],transform=ax.transAxes,fontsize=9)
    if j==0:ax.legend(fontsize=8,loc='upper left')
fig.text(.055,.022,'Decision: no new SplatAD fit. Both real baselines fail the frozen ADE ≤ 1 m / FDE ≤ 2 m requirement.\nThese are 16 policy forwards + vehicle forecasts, not feedback loops; route hint is extra ground-truth information. No reconstruction was tested here.',fontsize=12,color='#344858')
stem=a.out/'F21_Fresh_Real_Baseline_Control'
for ext in ['png','pdf','svg']:
    p=stem.with_suffix('.'+ext);fig.savefig(p,dpi=190)
    if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
print('FIGURE',stem)
