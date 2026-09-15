import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;D=W/'evidence_milestone3';O=W.parents[1]/'outputs/Simulation_Impact_Research'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
rows=json.loads((D/'pdm_simulation_summary.json').read_text());base={r['scene']:r for r in rows if r['condition']=='real'};names=sorted(base)
methods=['vggt','omega512','dvgt1','pi3x'];labels=['VGGT','VGGT-Ω 512','DVGT-1','Pi3X'];ind=[(m,v) for m in methods for v in ['six','twelve']]
A=np.zeros((8,6));S=A.copy();C=A.copy()
for i,(m,v) in enumerate(ind):
 for j,n in enumerate(names):
  r=next(r for r in rows if r['scene']==n and r.get('method')==m and r.get('variant')==v and r.get('protocol')=='build_scale' and r['condition']=='full')
  A[i,j]=r['ADE_vs_recorded_ego_m']-base[n]['ADE_vs_recorded_ego_m'];S[i,j]=r['final_position_change_vs_real_m'];C[i,j]=int(r['actor_overlap_any'])-int(base[n]['actor_overlap_any'])
fig,axs=plt.subplots(1,2,figsize=(14,6.5));limits=[max(1.,S.max()),max(.15,abs(A).max())]
for ax,data,title,cmap,lo,hi,fmt in [(axs[0],S,'Executed endpoint change (m)','Blues',0,limits[0],'.2f'),(axs[1],A,'Change in recorded-path ADE (m)','RdBu_r',-limits[1],limits[1],'+.2f')]:
 im=ax.imshow(data,cmap=cmap,vmin=lo,vmax=hi,aspect='auto');ax.set_title(title,fontsize=13,pad=12)
 ax.set_xticks(range(6),[n[-4:] for n in names]);ax.set_yticks(range(8),[f'{labels[methods.index(m)]} · {6 if v=="six" else 12}' for m,v in ind] if ax==axs[0] else ['']*8);ax.tick_params(length=0)
 for (i,j),x in np.ndenumerate(data):
  ink='white' if (cmap=='Blues' and x>.78*hi) or (cmap=='RdBu_r' and abs(x)>.7*hi) else 'black'
  ax.text(j,i,format(x,fmt),ha='center',va='center',fontsize=10,color=ink)
  if C[i,j]>0:ax.plot(j+.30,i-.30,'*',c='#ad251b',ms=10)
 fig.colorbar(im,ax=ax,shrink=.8,pad=.03)
full=[r for r in rows if r.get('protocol')=='build_scale' and r['condition']=='full'];novel=sum(r['actor_overlap_any'] and not base[r['scene']]['actor_overlap_any'] for r in full)
fig.suptitle('F08  Natural interaction cohort: ordinary scale control and complete denominators',x=.035,ha='left',fontsize=18,weight='bold')
fig.text(.035,.87,f'48 full scans · 6 distinct logs · known camera calibration + one BUILD-LiDAR scalar · {novel} new actor overlaps',fontsize=11,color='#53657b')
fig.text(.035,.035,'Negative ADE change = closer to the recorded ego path. Stars denote overlap absent with real LiDAR under the same state.\nOfficial TransFuser + official PDM 4 s tracking, nuScenes adapter; actor overlap is not a complete PDM safety score.',fontsize=11,color='#53657b')
fig.subplots_adjust(left=.12,right=.97,top=.79,bottom=.18,wspace=.25)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F08_Interaction_Controlled_Outcomes.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
print('Saved F08; new overlap',novel)
