"""把真实接触对象与几何余量放在图中，避免把路锥交叠画成撞车。"""
import json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from PIL import Image
W=Path(__file__).resolve().parent;D=W/'evidence_milestone3';P=D/'scene-0061';O=W.parents[1]/'outputs/Simulation_Impact_Research'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#267caa';red='#cf4946';gray='#53657b';orange='#df922e'
ref=json.loads((P/'log_reference.json').read_text());inp=json.loads((P/'input.json').read_text());audit=json.loads((D/'clearance_audit.json').read_text());pol=[r for r in json.loads((D/'policy_probe_summary.json').read_text()) if r['scene']=='scene-0061'];s=np.load(P/'states.npz');tracked=s['simulated'];time=s['times'];ib=next(i for i,r in enumerate(pol) if r['condition']=='real');ip=next(i for i,r in enumerate(pol) if r.get('method')=='pi3x' and r.get('variant')=='six' and r.get('protocol')=='build_scale' and r['condition']=='full')
def rect(x,y,w,l,yaw):
 a=np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]]);r=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);return a@r.T+[x,y]
actors=[a for a in ref['records'][0]['boxes'] if a['category']=='movable_object.trafficcone' and 0<a['box'][0]<22 and abs(a['box'][1])<5]
fig=plt.figure(figsize=(15,8));gs=fig.add_gridspec(2,3,width_ratios=[1.1,1,1.15],height_ratios=[1,1],wspace=.35,hspace=.35)
ax=fig.add_subplot(gs[0,0:2]);im=np.array(Image.open(P/'front.jpg'));v=inp['views'][0];cam=np.linalg.inv(np.array(v['world_from_ego_camera']))@np.array(v['world_from_camera']);K=np.array(v['intrinsics_original']);sc=im.shape[1]/v['original_wh'][0];ax.imshow(im)
for a in actors:
 b=np.array(a['box']);y=b[6];rot=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]]);c=np.array(list(itertools.product([-1,1],repeat=3)))*np.array([b[4],b[3],b[5]])/2;c=c@rot.T+b[:3];q=(c-cam[:3,3])@cam[:3,:3];uv=q@K.T;uv=uv[:,:2]/uv[:,2:]*sc;lo=uv.min(0);hi=uv.max(0);ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,ec='#f3cb4e',lw=2))
ax.set_ylim(im.shape[0],im.shape[0]*.28);ax.set_xlim(im.shape[1]*.12,im.shape[1]*.95);ax.axis('off');ax.set_title('(a) Real RGB: the queried objects are traffic cones',loc='left',fontsize=13,weight='bold')
ax=fig.add_subplot(gs[:,2]);
for a in actors:
 b=a['box'];pts=rect(b[0],b[1],b[3],b[4],b[6]);ax.add_patch(Polygon(pts[:,[1,0]],fc=orange,ec='#9c621c',alpha=.8))
ax.plot(s['ground_truth_ego'][:,1],s['ground_truth_ego'][:,0],'--',c=gray,label='Recorded ego')
for idx,c,l in [(ib,blue,'Real LiDAR'),(ip,red,'Pi3X full reconstruction')]:
 ax.plot(tracked[idx,:,1],tracked[idx,:,0],c=c,label=l,lw=2)
 for j in [25,40]:
  p=tracked[idx,j];vp=audit['ego_parameters'];x=p[0]+vp['rear_axle_to_center']*np.cos(p[2]);y=p[1]+vp['rear_axle_to_center']*np.sin(p[2]);q=rect(x,y,vp['width'],vp['length'],p[2]);ax.add_patch(Polygon(q[:,[1,0]],fill=False,ec=c,lw=1.2,alpha=.8))
ax.set(xlim=(3.4,-1.5),ylim=(-1,19),xlabel='Left (m)',ylabel='Forward (m)');ax.set_title('(c) Official vehicle execution',loc='left',fontsize=13,weight='bold');ax.legend(loc='lower right',fontsize=9)
ax.text(.04,.95,'Outlines at 2.5 s and 4.0 s\nOrange = recorded cone boxes',transform=ax.transAxes,va='top',fontsize=10)
ax=fig.add_subplot(gs[1,:2]);selected=[r for r in audit['executions'] if r['condition']=='real' or (r.get('protocol')=='build_scale' and r['condition']=='full')];names=[];nom=[];low=[];high=[]
for r in selected:
 names.append('Real\nLiDAR' if r['condition']=='real' else {'dvgt1':'DVGT-1','vggt':'VGGT','omega512':'Ω 512','pi3x':'Pi3X'}[r['method']]+'\n'+('6' if r['variant']=='six' else '12'))
 z=r['clearance'];nom.append(z['ego_margin_+0.0m']['signed_clearance_m']);low.append(z['ego_margin_+0.1m']['signed_clearance_m']);high.append(z['ego_margin_-0.1m']['signed_clearance_m'])
x=np.arange(len(names));ax.axhline(0,c=gray,lw=1);ax.vlines(x,low,high,color=gray,lw=2);ax.scatter(x,nom,c=[blue]+[red]*8,s=40,zorder=3);ax.set_xticks(x,names,fontsize=10);ax.set_ylabel('Minimum signed clearance (m)');ax.set_title('(b) Contact is real in the metric; its interpretation needs a margin check',loc='left',fontsize=12,weight='bold');ax.text(.01,.07,'Vertical lines: ±0.1 m ego footprint margin\nNegative values: rectangle penetration',transform=ax.transAxes,fontsize=10,color=gray)
fig.suptitle('F09  A natural downstream candidate — cone contact, not a demonstrated severe vehicle crash',x=.035,ha='left',fontsize=17,weight='bold')
fig.text(.035,.015,'Real-LiDAR baseline clearance is only 0.016 m; its nominal safe label is fragile. Recorded ego has 0.495 m minimum clearance.\nAll eight geometry cases use one BUILD-LiDAR scale control. Nonreactive 4 s tracking and annotation/vehicle conventions limit the claim.',fontsize=11,color=gray)
fig.subplots_adjust(top=.90,bottom=.13)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F09_Cone_Contact_Candidate.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
print('Saved F09')
