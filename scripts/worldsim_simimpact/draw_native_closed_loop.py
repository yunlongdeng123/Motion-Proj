"""真实场景、实际反馈轨迹和模态对照；不把试跑当作确认的几何坏例。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
W=Path(__file__).resolve().parent;D=W/'evidence_native_pilot';O=W.parents[1]/'outputs/Simulation_Impact_Research'
q=json.loads((D/'downstream_audit.json').read_text());ref=json.loads((D/'log_reference.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(16,9))
g=fig.add_gridspec(2,3,left=.045,right=.98,top=.90,bottom=.13,wspace=.34,hspace=.36,height_ratios=[1,1.08],width_ratios=[1.05,1.05,1])
for col,name,title in [(0,'real_front_initial.jpg','(a) Real input at the initial pose'),(1,'render_front_initial.jpg','(b) SplatAD, 8k-step asset')]:
    ax=fig.add_subplot(g[0,col]);ax.imshow(Image.open(D/name));ax.axis('off');ax.set_title(title,loc='left',weight='bold',fontsize=12)
ax=fig.add_subplot(g[0,2]);ax.axis('off')
ax.text(0,1,'Actual feedback loop',va='top',weight='bold',fontsize=13,color='#173c61')
ax.text(0,.85,'Executed ego pose\n       ↓\nSplatAD RGB + LiDAR\n       ↓\nFrozen TransFuser\n       ↓\nPDM controller + bicycle\n       ↳ new ego pose',va='top',fontsize=12,linespacing=1.12)
ax.text(0,.03,'Replan: 0.5 s  ·  Physics: 0.1 s\nTwo 4 s rollouts from the same start',fontsize=10,color='#526477')

ax=fig.add_subplot(g[1,0]);tt=np.arange(41)*.1
rt=np.array([r['time_s'] for r in ref['records']]);rxy=np.array([r['xy_yaw'] for r in ref['records']])
gt=np.c_[np.interp(tt,rt,rxy[:,0]),np.interp(tt,rt,rxy[:,1])]
ax.plot(gt[:,0],gt[:,1],color='#3b4855',lw=2,label='Recorded ego')
colors={'closedloop_raw':'#2369a1','closedloop_median':'#ce772d'}
for condition,color in colors.items():
    s=np.load(D/condition/'executed_states.npz')['states'];ax.plot(s[:,0],s[:,1],color=color,lw=2,label=condition.replace('closedloop_','Native '));ax.scatter(s[-1,0],s[-1,1],s=35,color=color)
ax.scatter(gt[-1,0],gt[-1,1],s=35,color='#3b4855')
ax.set(xlabel='Forward displacement (m)',ylabel='Lateral displacement (m)',ylim=(-1.15,.8),xlim=(-.3,24))
ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper left');ax.set_title('(c) Executed trajectories',loc='left',weight='bold',fontsize=12)
ax.text(.02,.02,'Axes have different scales.\nRecorded-ego mismatch is not\nall caused by reconstruction.',transform=ax.transAxes,fontsize=9,color='#526477')

ax=fig.add_subplot(g[1,1]);styles=[('rendered_sensor_teacher_forcing_raw','Both: raw','#2369a1','-'),('rendered_sensor_teacher_forcing_median','Both: median','#ce772d','-'),('lidar_only_teacher_forcing_raw','LiDAR only: raw','#2369a1','--'),('lidar_only_teacher_forcing_median','LiDAR only: median','#ce772d','--'),('rgb_only_teacher_forcing','RGB only','#7a548e',':')]
for condition,label,color,ls in styles:
    rows=sorted([r for r in q['replay'] if r['condition']==condition],key=lambda r:r['index'])
    ax.plot([r['time_s'] for r in rows],[r['ADE_delta_vs_real_m'] for r in rows],label=label,color=color,ls=ls,lw=1.7,marker='o',markersize=3)
ax.axhline(0,color='#333',lw=.7);ax.grid(alpha=.2)
ax.set(xlabel='Logged query time (s)',ylabel='Execution ADE change vs real sensors (m)')
ax.set_title('(d) Same-pose modality controls',loc='left',weight='bold',fontsize=12)
ax.legend(loc='upper left',bbox_to_anchor=(0,1.34),ncol=2,fontsize=8,frameon=False);ax.text(.99,.01,'Positive = worse agreement\nwith recorded ego',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#526477')

ax=fig.add_subplot(g[1,2]);ax.axis('off');ax.text(0,1,'Measured consequences',va='top',weight='bold',fontsize=13,color='#173c61')
lines=[]
for row,label in zip(q['closed_loop'],['Raw','Median']):
    lines.append(f"{label}: ADE {row['ADE_vs_recorded_ego_m']:.2f} m\n      minimum box gap {row['clearance']['0.0']['signed_box_clearance_m']:.2f} m")
ax.text(0,.83,'\n\n'.join(lines)+'\n\nNo annotated-box overlap\nin either 4 s rollout.',va='top',fontsize=11,linespacing=1.3)
ax.text(0,.17,'Interpretation\nRGB-only replacement reproduces\nthe replay gap. LiDAR-only effects\nare small in this pilot.',va='top',fontsize=10,weight='bold',color='#173c61')
fig.suptitle('Native sensor feedback is running; the replay gap mainly follows the RGB channel',x=.02,ha='left',weight='bold',fontsize=17)
fig.text(.02,.066,'scene-0004 · 8k / 30k-step integration pilot · 64 policy/controller forecasts · 31 eligible vehicle observations (not unique actors)',fontsize=10,color='#526477')
fig.text(.02,.038,'RGB-channel attribution does not distinguish geometry from appearance. No local geometry repair or independent confirmation yet.',fontsize=10,color='#526477')
for ext in ['png','pdf','svg']:fig.savefig(O/f'F13_Native_RGB_LiDAR_Loop.{ext}',dpi=180,bbox_inches='tight')
