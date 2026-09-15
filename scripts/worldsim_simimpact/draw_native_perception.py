"""用完整 RGB 场景、模块数据流、实际扫描和检测结果呈现候选。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyBboxPatch,Polygon
from PIL import Image
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent;D=W/'evidence_native_objects';A=W/'evidence_native_full';O=W.parents[1]/'outputs/Simulation_Impact_Research'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
objects=json.loads((D/'objects.json').read_text());target=next(x for x in objects if x['index']==0 and x['class']=='construction_vehicle')
projection=next(x for x in json.loads((D/'frame_00_CAM_FRONT_projection.json').read_text()) if x['instance']==target['instance'])
fig=plt.figure(figsize=(16,10));g=fig.add_gridspec(3,6,left=.05,right=.98,top=.91,bottom=.13,height_ratios=[1.2,.25,1],wspace=.48,hspace=.3)
for slot,file,title in [(g[0,:3],D/'frame_00_CAM_FRONT.jpg','(a) Real log: excavator behind the barriers'),(g[0,3:],A/'render_front_initial.jpg','(b) SplatAD at the same camera pose')]:
    ax=fig.add_subplot(slot);ax.imshow(Image.open(file));ax.axis('off');ax.set_title(title,loc='left',weight='bold',fontsize=12)
    x0,y0,x1,y1=projection['xyxy'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffce32',lw=2))
    ax.text(x0,y0-12,'GT object extent',color='#ffce32',fontsize=9,weight='bold',bbox={'facecolor':'#17232a','alpha':.75,'edgecolor':'none','pad':3})
ax=fig.add_subplot(g[1,:]);ax.axis('off')
labels=['Full RGB + LiDAR logs','SplatAD scene asset','10-sweep simulated LiDAR','Frozen CenterPoint','GT object matches']
for j,label in enumerate(labels):
    x=j*.205;ax.add_patch(FancyBboxPatch((x,.22),.17,.6,boxstyle='round,pad=.01',fc='#edf3f7',ec='#7593ad'))
    ax.text(x+.085,.52,label.replace(' simulated','\nsimulated').replace(' scene','\nscene').replace(' logs','\nlogs'),ha='center',va='center',fontsize=10)
    if j<4:ax.annotate('',xy=(x+.20,.52),xytext=(x+.176,.52),arrowprops={'arrowstyle':'->','color':'#385570','lw':1.4})
ax.set_xlim(-.02,1.015);ax.set_ylim(0,1)
z=np.load(D/'frame_00_points.npz');ext=z['lidar_to_ego'];quat=np.array(target['rotation']);rot=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();w,l,h=target['wlh'];corners=np.array([[l/2,w/2,0],[l/2,-w/2,0],[-l/2,-w/2,0],[-l/2,w/2,0]])@rot.T+target['center_lidar'];corners=corners@ext[:3,:3].T+ext[:3,3]
for col,c,title in [(0,'real','(c) Real LiDAR: detected'),(2,'raw','(d) Native raw: missed')]:
    ax=fig.add_subplot(g[2,col:col+2]);p=z[c];keep=(p[:,0]>16)&(p[:,0]<26)&(p[:,1]>4)&(p[:,1]<13)&(p[:,2]>.6)
    ax.scatter(p[keep,0],p[keep,1],c=p[keep,2],s=9,vmin=.6,vmax=3.8,cmap='viridis',rasterized=True)
    ax.add_patch(Polygon(corners[:,:2],fill=False,edgecolor='#c29b00',lw=1.6,ls='--'))
    ax.set(xlim=(16,26),ylim=(4,13),xlabel='Forward (m)',ylabel='Left (m)');ax.set_aspect('equal');ax.grid(alpha=.18);ax.set_title(title,loc='left',weight='bold',fontsize=12)
    ax.text(.02,.98,'Current scan; height > 0.6 m\nDashed: GT annotation footprint',transform=ax.transAxes,va='top',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
ax=fig.add_subplot(g[2,4:]);conditions=['real','raw','median','logged_raw','logged_median'];summary=json.loads((D/'instance_summary.json').read_text())[target['instance']]
vals=[summary[c] for c in conditions];labels=['Real','Native raw','Native median','Logged rays: raw','Logged rays: median'];yy=np.arange(5)
ax.barh(yy,vals,color=['#3c8165','#ba6353','#dba271','#ba6353','#dba271'],height=.62);ax.invert_yaxis();ax.set_yticks(yy,labels,fontsize=9);ax.set(xlim=(0,9.3),xticks=[0,2,4,6,8],xlabel='Matched observations / 8');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for y,v in zip(yy,vals):ax.text(v+.12,y,f'{v}/8',va='center',fontsize=10,weight='bold')
ax.set_title('(e) Repeated detection loss',loc='left',weight='bold',fontsize=12)
fig.suptitle('A visible excavator is repeatedly missed from reconstructed LiDAR',x=.035,ha='left',fontsize=18,weight='bold')
fig.text(.04,.075,'scene-0004 · full 30,001-step SplatAD fit · one actor at eight existing log times · class-aware BEV IoU ≥ 0.5, score ≥ 0.5',fontsize=10,color='#405a70')
fig.text(.04,.047,'The RGB panels locate the object; CenterPoint consumes LiDAR only. This is a sensor/perception candidate, not confirmed geometry causality or planning harm.',fontsize=10,color='#405a70')
for extn in ['png','pdf','svg']:fig.savefig(O/f'F15_Excavator_Perception_Candidate.{extn}',dpi=190,bbox_inches='tight')
plt.close(fig)

rows=json.loads((W/'evidence_native_detector/ff_summary.json').read_text());fig,axs=plt.subplots(1,2,figsize=(14,5));fig.subplots_adjust(left=.12,right=.98,top=.77,bottom=.27,wspace=.30)
conds=['real','vggt_full','vggt_fill_missing','omega512_full','omega512_fill_missing','dvgt1_full','dvgt1_fill_missing','pi3x_full','pi3x_fill_missing']
names=['Real','VGGT','VGGT + missing restored','VGGT-Ω 512','Ω + missing restored','DVGT-1','DVGT-1 + missing restored','Pi3X','Pi3X + missing restored']
for ax,scene in zip(axs,['scene-0004','scene-0061']):
    rr=[next(r for r in rows if r['scene']==scene and r['condition']==c) for c in conds];n=len(rr[0]['eligible_gt']);m=np.array([[r['scores']['0.5'][k]['matched'] for k in ['center2m','bev_iou0p5']] for r in rr])
    ax.imshow(m/n,aspect='auto',cmap='Blues',vmin=0,vmax=1);ax.set_yticks(range(len(names)),names if scene=='scene-0004' else ['']*len(names),fontsize=9);ax.set_xticks([0,1],['Center ≤ 2 m','BEV IoU ≥ 0.5']);ax.set_title(scene+f' · {n} eligible objects',weight='bold')
    for i in range(len(names)):
        for j in range(2):ax.text(j,i,f'{m[i,j]}/{n}',ha='center',va='center',color='white' if m[i,j]/n>.65 else '#17384e',fontsize=10)
fig.suptitle('Four feed-forward meshes: current-scan perception isolation',x=.03,ha='left',weight='bold',fontsize=17)
fig.text(.03,.86,'6 RGB views → existing metric-scale mesh → current scan + same 9 real past scans → frozen CenterPoint',fontsize=12,color='#244a67')
fig.text(.03,.16,'Known poses, BUILD LiDAR scale, current real intensity and real history are extra information. Missing restoration additionally uses real ranges.',fontsize=10)
fig.text(.03,.10,'Two exposed discovery logs · 18 detector calls · score ≥ 0.5 · all eligible categories included · match counts, not AP/NDS.',fontsize=10)
fig.text(.03,.045,'DVGT-1 improves IoU matches on 0004; losses in other rows do not establish a universal failure or complete-simulator degradation.',fontsize=10,color='#244a67')
for extn in ['png','pdf','svg']:fig.savefig(O/f'F16_Feedforward_Perception_Control.{extn}',dpi=190,bbox_inches='tight')
