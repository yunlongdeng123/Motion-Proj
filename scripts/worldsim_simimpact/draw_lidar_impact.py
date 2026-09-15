"""真实数据科学图：保留未造成严重后果的对照，不画虚构碰撞。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle
from PIL import Image
W=Path(__file__).resolve().parent;D=W/'evidence_milestone2';O=W.parents[1]/'outputs/Simulation_Impact_Research';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
blue='#267caa';red='#cf4946';green='#27856b';gray='#53657b';yellow='#f3c848'
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/f'{name}.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def box(ax,xy,w,h,text,color='#eaf1f6'):
 ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=.01,rounding_size=.025',facecolor=color,edgecolor='#9aaebd'))
 ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=10)
def arrow(ax,a,b,color=gray):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':color,'lw':1.8})

fig,ax=plt.subplots(figsize=(14,4.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
box(ax,(.01,.48),.15,.24,'6 / 12 real RGB\nRGB-only model input')
box(ax,(.21,.48),.16,.24,'Official geometry\nVGGT / Ω\nDVGT-1 / Pi3X')
box(ax,(.42,.48),.16,.24,'Mesh first intersection\nKnown poses / intrinsics\nMeasured beam directions')
box(ax,(.63,.48),.15,.24,'Official TransFuser\nRGB + LiDAR → plan')
box(ax,(.83,.48),.16,.24,'Official PDM simulator\nLQR + bicycle dynamics')
for a,b in [((.16,.6),(.21,.6)),((.37,.6),(.42,.6)),((.58,.6),(.63,.6)),((.78,.6),(.83,.6))]:arrow(ax,a,b)
box(ax,(.21,.08),.37,.19,'Real LiDAR baseline + one global-scale control\nEarly / late / missing oracle hybrids','#fff4dc')
arrow(ax,(.49,.28),(.49,.48));arrow(ax,(.58,.17),(.70,.48))
box(ax,(.68,.08),.31,.19,'Executed 4 s path, acceleration, actor overlap\nNo pose-responsive sensor replanning','#e7f2ed')
arrow(ax,(.91,.48),(.91,.28))
ax.text(.01,.91,'F03  Which geometry errors actually reach a LiDAR policy?',fontsize=19,weight='bold')
ax.text(.01,.81,'A controlled nuScenes adapter to official NAVSIM components; not a native benchmark leaderboard.',fontsize=12,color=gray)
save(fig,'F03_LiDAR_Architecture')

rows=json.loads((D/'pdm_simulation_summary.json').read_text());rays=json.loads((D/'raycast_summary.json').read_text());base={r['scene']:r for r in rows if r['condition']=='real'}
names=['scene-0013','scene-0038','scene-0041'];methods=['vggt','omega512','dvgt1','pi3x'];labels=['VGGT','VGGT-Ω 512','DVGT-1','Pi3X'];variants=['six','twelve']
index=[(m,v) for m in methods for v in variants];A=np.zeros((8,3));S=A.copy();E=A.copy()
for i,(m,v) in enumerate(index):
 for j,n in enumerate(names):
  r=next(r for r in rows if r.get('method')==m and r.get('variant')==v and r.get('protocol')=='build_scale' and r['condition']=='full' and r['scene']==n)
  q=next(q for q in rays if q['method']==m and q['variant']==v and q['protocol']=='build_scale' and q['scene']==n)
  A[i,j]=r['ADE_vs_recorded_ego_m']-base[n]['ADE_vs_recorded_ego_m'];S[i,j]=r['final_position_change_vs_real_m'];E[i,j]=q['early_0p2']/q['rays']*100
fig,axs=plt.subplots(1,3,figsize=(13,6.4),gridspec_kw={'width_ratios':[1,1,1]})
for ax,data,title,cmap,vmin,vmax,fmt in zip(axs,[E,S,A],['Early returns (%)','Executed endpoint change (m)','Change in recorded-path ADE (m)'],['YlOrRd','Blues','RdBu_r'],[0,0,-.15],[60,1.2,.15],['.1f','.2f','+.3f']):
 im=ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax,aspect='auto')
 ax.set_xticks(range(3),[n[-4:] for n in names]);ax.set_yticks(range(8),[f'{labels[methods.index(m)]} · {6 if v=="six" else 12}' for m,v in index] if ax==axs[0] else ['']*8)
 ax.set_title(title,fontsize=12,pad=13);ax.tick_params(length=0)
 for (i,j),x in np.ndenumerate(data):ax.text(j,i,format(x,fmt),ha='center',va='center',fontsize=10,color='black')
 fig.colorbar(im,ax=ax,shrink=.8,pad=.03)
fig.suptitle('F05  Many early returns; no new actor collision in this initial cohort',x=.03,ha='left',fontsize=18,weight='bold')
fig.text(.03,.88,'24 full scans after one global BUILD-LiDAR scale correction · 3 scenes · fixed RGB and vehicle state',fontsize=11,color=gray)
fig.text(.03,.04,'Negative ADE change = closer to the recorded path. Zero actor overlap in all 195 real/full/oracle-hybrid executions.\nThis is 4 s nonreactive tracking with a nuScenes adapter; it does not establish absence of harm in general.',fontsize=11,color=gray)
fig.subplots_adjust(left=.14,right=.97,top=.8,bottom=.18,wspace=.35);save(fig,'F05_LiDAR_Cohort_Controls')

n='scene-0013';m='omega512';P=D/n;ref=json.loads((P/'log_reference.json').read_text());inp=json.loads((P/'input.json').read_text());real=np.load(P/'real.npz');pred=np.load(P/f'{m}_six_build_scale.npz');gt=real['points'];origin=real['origin'];distance=pred['first_range'];delta=distance-real['ranges'];b=np.array(ref['records'][0]['boxes'][0]['box']);yaw=b[6];rot=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
local=(gt-b[:3])@rot;mask=(abs(local)<np.array([b[4],b[3],b[5]])/2+.1).all(1);early=mask&(delta<-.2);ids=np.where(early)[0];sel=ids[np.argmin(abs(delta[ids]-np.median(delta[ids])))];ph=origin+real['directions']*distance[:,None]
view=inp['views'][0];Tc=np.linalg.inv(np.array(view['world_from_ego_camera']))@np.array(view['world_from_camera']);K=np.array(view['intrinsics_original']);im=np.array(Image.open(P/'front.jpg'));scale=im.shape[1]/view['original_wh'][0]
def project(x):
 c=(x-Tc[:3,3])@Tc[:3,:3];uv=c@K.T;return uv[:,:2]/uv[:,2:]*scale
uv=project(gt[mask]);pv=project(ph[early]);x0,y0=uv.min(0)-8;x1,y1=uv.max(0)+8
pr=json.loads((D/'policy_probe_summary.json').read_text());scene_rows=[r for r in pr if r['scene']==n];ix=next(i for i,r in enumerate(scene_rows) if r.get('method')==m and r.get('variant')=='six' and r.get('protocol')=='build_scale' and r['condition']=='full');ib=next(i for i,r in enumerate(scene_rows) if r['condition']=='real');states=np.load(P/'states.npz');tracked=states['simulated'];gtpath=states['ground_truth_ego'];rr=next(r for r in rows if r['scene']==n and r.get('method')==m and r.get('variant')=='six' and r.get('protocol')=='build_scale' and r['condition']=='full')
fig=plt.figure(figsize=(15,8.7));gs=fig.add_gridspec(2,3,height_ratios=[1.15,1],hspace=.44,wspace=.34)
ax=fig.add_subplot(gs[0,:2]);ax.imshow(im);ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=yellow,lw=2));ax.scatter(pv[:,0],pv[:,1],s=6,c=red,alpha=.9);ax.set_xlim(200,1000);ax.set_ylim(560,130);ax.axis('off');ax.set_title('(a) Real RGB + projected earlier model returns',loc='left',fontsize=13,weight='bold')
ax.text(.01,.03,'Yellow: reference bus returns   Red: earlier model returns',transform=ax.transAxes,color='white',bbox={'facecolor':'black','alpha':.6,'pad':4},fontsize=10)
ax=fig.add_subplot(gs[0,2]);ax.axis('off');ax.text(0,.92,'VGGT-Ω original 512\n6 RGB + one LiDAR scale',fontsize=15,weight='bold',va='top')
ax.text(0,.60,f'{early.sum()} / {mask.sum()} bus-box returns\nare early by > 0.2 m.\n\nRepresentative beam:\n{float(-delta[sel]):.2f} m early',fontsize=13,va='top',linespacing=1.5)
ax.text(0,.06,'Measured points fall inside the bus annotation.\nThis alone does not identify opaque surfaces.\nSelected to show the sensor change, not an accident.',fontsize=10,color=gray,va='bottom')
ax=fig.add_subplot(gs[1,0]);ax.scatter(gt[mask,0],gt[mask,2],s=10,c=blue,label='Measured first return');ax.scatter(ph[early,0],ph[early,2],s=9,c=red,label='Reconstructed first return')
ax.plot([origin[0],gt[sel,0]],[origin[2],gt[sel,2]],color=gray,lw=1,alpha=.6);ax.scatter([origin[0]],[origin[2]],s=50,marker='>',color='black');ax.text(origin[0]+1,origin[2]+.2,'Sensor',fontsize=9)
ax.annotate(f'{-delta[sel]:.2f} m along ray',xy=(ph[sel,0],ph[sel,2]),xytext=(12,4.3),arrowprops={'arrowstyle':'->','color':red},color=red,fontsize=10)
ax.set(xlim=(-1,45),ylim=(-.5,5.1),xlabel='Forward (m)',ylabel='Height (m)');ax.set_title('(b) Actual first-return displacement',loc='left',fontsize=12,weight='bold');ax.legend(fontsize=8,loc='lower right')
ax=fig.add_subplot(gs[1,1]);h0=np.load(P/'real_histogram.npy')[0];h1=np.load(P/f'{m}_six_build_scale_full_histogram.npy')[0];diff=h1-h0;v=ax.imshow(diff,origin='lower',extent=(-32,32,-32,32),cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal');ax.set(xlim=(-15,15),ylim=(-5,32),xlabel='Left (m)',ylabel='Forward (m)');ax.set_title('(c) Policy BEV input changes',loc='left',fontsize=12,weight='bold');cb=fig.colorbar(v,ax=ax,fraction=.035,pad=.03);cb.ax.set_title('Δ input',fontsize=9)
ax=fig.add_subplot(gs[1,2]);ax.plot(gtpath[:,1],gtpath[:,0],'--',color=gray,label='Recorded ego');ax.plot(tracked[ib,:,1],tracked[ib,:,0],c=blue,label='Real LiDAR');ax.plot(tracked[ix,:,1],tracked[ix,:,0],c=red,label='Reconstructed LiDAR');ax.set(xlabel='Left (m)',ylabel='Forward (m)');ax.invert_xaxis();ax.legend(fontsize=9);ax.set_title('(d) Official 4 s vehicle tracking',loc='left',fontsize=12,weight='bold');ax.text(.03,.45,f'Endpoint shift: {rr["final_position_change_vs_real_m"]:.2f} m\nNew actor overlap: NO',transform=ax.transAxes,fontsize=10,bbox={'fc':'white','ec':'none','alpha':.9})
fig.suptitle('F04  RGB → earlier returns → BEV change → measured driving response',x=.04,ha='left',fontsize=19,weight='bold');fig.text(.04,.015,'Downstream uses the full reconstructed scan, not only the bus rays. Same RGB and state throughout; no collision or false-safe is inferred from a trajectory shift.',fontsize=11,color=gray)
fig.subplots_adjust(top=.91,bottom=.09);save(fig,'F04_Real_Sensor_To_Driving')

c=json.loads((D/'renderer_depth_contract.json').read_text());fig,ax=plt.subplots(figsize=(10.5,4.7));x=np.arange(3);w=.24
ax.bar(x-w,[r['RGB_ED_S_raw_depth_channel'] for r in c],w,color=red,label='RGB+ED+S raw depth channel')
ax.bar(x,[r['RGB_ED_S_divided_by_alpha_m'] for r in c],w,color=green,label='Raw / accumulated opacity')
ax.bar(x+w,[r['RGB_ED_depth_m'] for r in c],w,color=blue,label='RGB+ED official normalized mode')
ax.axhline(10,c=gray,ls='--',lw=1);ax.set(xticks=x,xticklabels=[str(r['opacity']) for r in c],xlabel='Single-Gaussian opacity',ylabel='Reported depth (m)',ylim=(0,12.5));ax.legend(loc='upper left',fontsize=10,ncol=1)
for i,r in enumerate(c):ax.text(i-w,r['RGB_ED_S_raw_depth_channel']+.15,f'{r["RGB_ED_S_raw_depth_channel"]:.2f}',ha='center',fontsize=10)
fig.suptitle('F06  A depth readout mismatch can masquerade as an early surface',x=.04,ha='left',fontsize=17,weight='bold');fig.text(.04,.02,'GPU calibration: one Gaussian centered at 10 m. Installed official HUGSIM fork does not normalize RGB+ED+S.\nThis diagnoses the depth interface; it does not explain LTF rollouts, because that policy ignores the depth channel.',fontsize=10,color=gray)
fig.subplots_adjust(top=.83,bottom=.24);save(fig,'F06_Depth_Readout_Contract')
print('Saved F03–F06',O)
