"""RGB语境＋预先指定空间区域＋完整恢复结果，明确不是已确认Hero badcase。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
W=Path(__file__).resolve().parent;D=W/'evidence_pi3x_local';O=W.parents[1]/'outputs/Simulation_Impact_Research'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
z=np.load(D/'points.npz');mask=np.load(D/'region_masks.npz')['six_forward_right'];views=json.loads((D/'views.json').read_text());rows=json.loads((D/'localization_outcomes.json').read_text())['rows']
fig=plt.figure(figsize=(16,8.2));top=fig.add_gridspec(1,3,left=.04,right=.97,top=.88,bottom=.49,width_ratios=[1.5,.95,1.15],wspace=.32);bottom=fig.add_gridspec(1,3,left=.25,right=.97,top=.39,bottom=.09,width_ratios=[1.5,.95,1.15],wspace=.32)
a=fig.add_subplot(top[0,:2]);v=next(v for v in views if v['camera']=='CAM_FRONT_RIGHT');a.imshow(Image.open(D/'CAM_FRONT_RIGHT.jpg'))
T=np.linalg.inv(np.array(views[0]['world_from_ego_camera']))@np.array(v['world_from_camera']);cp=(z['gt'][mask]-T[:3,3])@T[:3,:3];uv=cp@np.array(v['intrinsics_original']).T;uv=uv[:,:2]/uv[:,2:];uv*=np.array(v['export_wh'])/np.array(v['original_wh']);valid=(cp[:,2]>.2)&(uv[:,0]>=0)&(uv[:,0]<v['export_wh'][0])&(uv[:,1]>=0)&(uv[:,1]<v['export_wh'][1])
a.scatter(*uv[valid][::3].T,s=5,c='#ffdf35',alpha=.6,lw=0);a.set_axis_off();a.set_title('(a) Real RGB: right-forward measured region',loc='left',weight='bold',fontsize=13)
a.text(.02,.03,'Yellow: projected measured endpoints in the selected region',transform=a.transAxes,color='white',fontsize=10,bbox={'facecolor':'#142b44','alpha':.85,'edgecolor':'none','pad':4})
b=fig.add_subplot(top[0,2]);p=z['gt'];b.scatter(-p[::7,1],p[::7,0],s=1,c='#98a6b4',alpha=.5);q=z['six'];vis=mask&(p[:,0]>-2)&(p[:,0]<30);b.scatter(-q[vis,1],q[vis,0],s=3,c='#d55453',alpha=.35,label='Pi3X return');b.add_patch(Rectangle((3,0),9,25,fill=False,edgecolor='#d19b15',lw=2));b.scatter([0],[0],marker='^',c='#173c61',s=55);b.set(xlim=(-14,15),ylim=(-3,29),xlabel='Right (m)',ylabel='Forward (m)');b.set_aspect('equal');b.set_title('(b) Fixed spatial region',loc='left',weight='bold',fontsize=13);b.legend(loc='upper left',fontsize=9)
c=fig.add_subplot(bottom[0,:2]);conditions=['full_restore_all_missing','repair_forward_corridor','repair_forward_left','repair_forward_right','repair_near_rear','repair_outside_local_regions'];labels=['Residual: all missing already restored','Restore forward corridor','Restore forward left','Restore forward right','Restore near rear','Restore outside regions (nonlocal)']
vals=np.array([[next(r for r in rows if r.get('variant')==v and r['condition']==co)['clearance']['0.0']['signed_clearance_m'] for v in ['six','twelve']] for co in conditions])
c.imshow(vals,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-.45,vcenter=0,vmax=.1),aspect='auto');c.set_xticks([0,1],['Pi3X: 6 views','Pi3X: 12 views']);c.set_yticks(range(len(labels)),labels);c.tick_params(length=0);c.set_title('(c) Minimum annotated-box clearance after sensor repair (m)',loc='left',weight='bold',fontsize=13,pad=14)
for y in range(len(labels)):
 for x in range(2):c.text(x,y,f'{vals[y,x]:+.3f}',ha='center',va='center',weight='bold',color='white' if vals[y,x]<-.2 else '#172b40')
d=fig.add_subplot(bottom[0,2]);d.axis('off');d.text(0,1,'Causal status',va='top',weight='bold',fontsize=14,color='#173c61')
d.text(0,.83,'6-view right-region repair:\nnominal contact disappears.\nOnly right-region error: contact.\n\n12-view same repair:\ncontact remains.\n\nReal baseline clearance: +0.016 m.\nThis is a fragile contact boundary,\nnot demonstrated serious harm.',va='top',fontsize=11,linespacing=1.45,color='#31495f')
fig.suptitle('Pi3X residual audit: local sensor recovery does not repeat across both inputs',x=.02,ha='left',fontsize=19,weight='bold')
fig.text(.02,.018,'All 23 conditions retained. CPU vs previous GPU policy difference < 4.2e-6. Sensor-only oracle is not geometry-asset repair; novel-pose loop remains pending.',fontsize=10,color='#5b6574')
for ext in ['png','pdf','svg']:fig.savefig(O/f'F12_Pi3X_Local_Causal_Audit.{ext}',dpi=180,bbox_inches='tight')
