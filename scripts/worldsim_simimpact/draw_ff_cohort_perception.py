"""展示完整下游分母、覆盖控制及原先按元数据冻结的前车；保留正例。"""
import json,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
W=Path(__file__).resolve().parent;E=W/'evidence_ff_cohort';E.mkdir(exist_ok=True)
with tarfile.open(W/'ff_cohort_evidence.tar.gz') as t:t.extractall(E,filter='data')
O=W.parents[1]/'outputs/Simulation_Impact_Research'
rows=json.loads((E/'scene_counts.json').read_text());reg=json.loads((E/'registration.json').read_text());objects=json.loads((E/'objects.json').read_text())
scenes=reg['scenes'];methods=['vggt','omega512','dvgt1','pi3x'];names=['VGGT','VGGT-Ω','DVGT-1','Pi3X'];columns=[(m,v) for m in methods for v in ['six','twelve']]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(16,9.2));grid=fig.add_gridspec(2,2,left=.07,right=.92,top=.79,bottom=.15,hspace=.55,wspace=.18,height_ratios=[1.2,1])
for j,condition in enumerate(['full','fill_missing']):
    ax=fig.add_subplot(grid[0,j]);values=[];texts=[]
    for scene in scenes:
        vv=[];tt=[]
        for m,v in columns:
            r=next(r for r in rows if r['scene']==scene and r['condition']==f'{m}_{v}_{condition}')
            loss=r['counts']['0.5']['bev_iou0p5']['lost_from_qualified_real'];n=r['real_qualified'];vv.append(loss/max(n,1));tt.append(f'{loss}/{n}')
        values.append(vv);texts.append(tt)
    im=ax.imshow(values,cmap='YlOrRd',vmin=0,vmax=1,aspect='auto')
    for y in range(6):
        for x in range(8):ax.text(x,y,texts[y][x],ha='center',va='center',fontsize=10,color='white' if values[y][x]>.6 else '#273640')
    ax.set_xticks(range(8),[n+'\n'+v for n in names for v in ['6 views','12 views']],fontsize=9)
    ax.set_yticks(range(6),[s.replace('scene-','') for s in scenes]);ax.tick_params(length=0)
    ax.set_title(('(a) Original reconstructed current scan' if j==0 else '(b) Restore all missing current returns'),loc='left',fontsize=12,weight='bold')
    ax.set_ylabel('Frozen log window' if j==0 else '')
cax=fig.add_axes([.935,.486,.011,.304]);fig.colorbar(im,cax=cax,label='Fraction of real-qualified objects losing IoU match')
ax=fig.add_subplot(grid[1,0]);vals=[];texts=[]
for scene in scenes:
    lead=reg['selection'][scene]['metadata_selection']['leads'][0]['instance'];vv=[];tt=[]
    for m,v in columns:
        r=next((r for r in objects if r['scene']==scene and r['instance']==lead and r['condition']==f'{m}_{v}_fill_missing'),None)
        if r is None or not r['baseline_qualified']:vv.append(np.nan);tt.append('N/A');continue
        center=r['scores']['0.5']['center_match'];iou=r['scores']['0.5']['IoU_match'];vv.append(0 if center and iou else (1 if center else 2));tt.append('OK' if center and iou else ('IoU' if center else 'Center'))
    vals.append(vv);texts.append(tt)
from matplotlib.colors import ListedColormap
cmap=ListedColormap(['#e0eee7','#f3d9b6','#d68774']);cmap.set_bad('#e4e7e9')
ax.imshow(np.ma.masked_invalid(vals),cmap=cmap,vmin=0,vmax=2,aspect='auto')
for y in range(6):
    for x in range(8):ax.text(x,y,texts[y][x],ha='center',va='center',fontsize=9,color='#273640')
ax.set_xticks(range(8),[n+'\n'+v for n in names for v in ['6','12']],fontsize=9);ax.set_yticks(range(6),[s.replace('scene-','') for s in scenes]);ax.tick_params(length=0)
ax.set_title('(c) Metadata-selected lead actor, after missing-return control',loc='left',weight='bold',fontsize=11)
ax=fig.add_subplot(grid[1,1]);ax.axis('off')
text='Cells in (a,b): lost / real-qualified objects.\nQualified: real scan matches class, center ≤2m\nand BEV IoU ≥0.5 at confidence ≥0.5.\n\n(c) OK: both matches retained; IoU: only overlap\nmatch lost; Center: center match lost. N/A:\nreal baseline is not qualified. No tuning by model.\n\nOne timestamp per exposed log. Point intensity,\nknown metric scale and nine real past sweeps\nare shared extra inputs. This is current-scan\nisolation, not a complete sensor simulation.'
ax.text(0,1,text,va='top',fontsize=10.5,linespacing=1.3,color='#30485c')
fig.suptitle('Which reconstruction errors survive into a frozen perception system?',x=.04,ha='left',fontsize=17,weight='bold')
aa=fig.add_axes([.04,.845,.91,.065]);aa.axis('off');labels=['6 / 12 RGB views','Four existing meshes','Current LiDAR raycast','Shared real history','Frozen CenterPoint']
for j,label in enumerate(labels):
    xx=j*.202;aa.add_patch(FancyBboxPatch((xx,.2),.17,.65,boxstyle='round,pad=.007',fc='#edf3f7',ec='#7998af'));aa.text(xx+.085,.525,label,ha='center',va='center',fontsize=10)
    if j<4:aa.annotate('',xy=(xx+.197,.525),xytext=(xx+.175,.525),arrowprops={'arrowstyle':'->','color':'#355772'})
fig.text(.05,.083,'Six distinct exposed logs · 102 paired matrix entries · 82 new detector forwards; 20 reused · No geometry model reruns.',color='#40586c',fontsize=10)
fig.text(.05,.045,'Detection localization / overlap loss is not necessarily an absent output, phantom surface, braking error or closed-loop collision.',color='#80513c',fontsize=10)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F19_Six_Log_Perception_Matrix.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
for p in O.glob('F19_*.svg'):p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
print('FIGURE_COMPLETE')
