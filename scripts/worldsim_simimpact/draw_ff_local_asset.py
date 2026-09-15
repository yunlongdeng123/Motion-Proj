"""真实场景、共享网格修改、完整扫描重投射和连续检测定位误差。"""
import json,tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,FancyBboxPatch
from scipy.spatial.transform import Rotation
from PIL import Image
W=Path(__file__).resolve().parent;D=W/'evidence_ff_local_asset_full';D.mkdir(exist_ok=True);O=W.parents[1]/'outputs/Simulation_Impact_Research'
with tarfile.open(W/'local_asset_evidence_r2.tar.gz') as t:t.extractall(D,filter='data')
support=json.loads((D/'support.json').read_text());rows=json.loads((D/'target_results.json').read_text());car=next(r for r in support if r['method']=='omega512');z=np.load(D/'omega512_points.npz');T=np.linalg.inv(z['lidar_from_ego'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(16,9.5));g=fig.add_gridspec(2,3,left=.055,right=.98,top=.90,bottom=.15,hspace=.35,wspace=.30,height_ratios=[1.1,1])
ax=fig.add_subplot(g[0,:2]);ax.imshow(Image.open(D/'omega512_context.jpg'));x0,y0,x1,y1=car['image_box_xyxy'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#ffd02c',lw=2))
ax.text(x0,y0-15,'Target vehicle: 25 real returns',color='#ffdc42',weight='bold',fontsize=9,bbox={'facecolor':'#17232a','edgecolor':'none','alpha':.8,'pad':3});ax.axis('off');ax.set_title('(a) Real driving scene and evaluated vehicle',loc='left',weight='bold',fontsize=12)
ax=fig.add_subplot(g[0,2]);ax.axis('off');labels=['6 / 12 RGB views → VGGT-Ω','Saved local mesh patch','Recast the complete current scan','Same 9 real past sweeps','Frozen CenterPoint → object box']
for j,label in enumerate(labels):
    yy=.93-j*.165;ax.add_patch(FancyBboxPatch((.02,yy-.08),.94,.12,boxstyle='round,pad=.015',fc='#edf3f7',ec='#7795ad'));ax.text(.49,yy-.02,label,ha='center',va='center',fontsize=10)
    if j<4:ax.annotate('',xy=(.49,yy-.132),xytext=(.49,yy-.09),arrowprops={'arrowstyle':'->','color':'#385570'})
ax.text(.02,.08,'Only mesh geometry changes; current intensity,\nmetric scale and real history stay fixed.',va='top',fontsize=9,color='#405a70')
ax.set_title('(b) Actual asset intervention',loc='left',weight='bold',fontsize=12)

ax=fig.add_subplot(g[1,0]);styles=[('gt','#283943','Real LiDAR'),('six_original','#c7664c','Original rendered'),('six_local_radial','#34866c','After local correction')]
for name,color,label in styles:
    p=z[name];ax.scatter(p[:,0],p[:,1],s=12,c=color,label=label,alpha=.85)
def corners(center,wlh,quat):
    w,l,h=wlh;q=np.array(quat);r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();p=np.array([[l/2,w/2,0],[l/2,-w/2,0],[-l/2,-w/2,0],[-l/2,w/2,0]])@r.T+center
    return (p@T[:3,:3].T+T[:3,3])[:,:2]
polys=[]
for condition,color in [('original','#c7664c'),('local_radial','#34866c')]:
    p=next(r for r in rows if r['method']=='omega512' and r['variant']=='six' and r['condition']==condition)['scores']['0.3']['nearest_same_class_prediction'];xy=corners(p['center_lidar'],p['wlh'],p['rotation']);polys.append(xy);ax.add_patch(Polygon(xy,fill=False,ec=color,lw=1.7));cc=np.mean(xy,0);ax.scatter(*cc,marker='x',c=color,s=50)
gt=car['gt'];xy=corners(gt['center_lidar'],gt['wlh'],gt['rotation']);polys.append(xy);ax.add_patch(Polygon(xy,fill=False,ec='#283943',ls='--',lw=1.4));boxall=np.concatenate(polys);ax.set(xlim=(boxall[:,0].min()-.8,boxall[:,0].max()+.8),ylim=(boxall[:,1].min()-1.4,boxall[:,1].max()+1.4),xlabel='Forward (m)',ylabel='Left (m)');ax.set_aspect('equal');ax.grid(alpha=.18);ax.legend(loc='upper left',fontsize=8)
ax.set_title('(c) Sensor geometry → box location',loc='left',weight='bold',fontsize=11)
ax.text(.02,.01,'Dashed box: GT annotation\nSolid boxes: actual detector predictions',transform=ax.transAxes,fontsize=8,color='#405a70')

ax=fig.add_subplot(g[1,1]);conds=['original','local_radial','same_faces_deleted'];labels=['Original','Local correction','Same-face deletion'];colors=['#c7664c','#34866c','#8c78a6']
for j,(condition,label,color) in enumerate(zip(conds,labels,colors)):
    yy=[next(r for r in rows if r['method']=='omega512' and r['variant']==v and r['condition']==condition)['scores']['0.3']['nearest_same_class_prediction']['center_error_m'] for v in ['six','twelve']];xx=np.arange(2)+(j-1)*.24
    ax.bar(xx,yy,width=.23,label=label,color=color)
    for x,y in zip(xx,yy):ax.text(x,y+.05,f'{y:.2f}',ha='center',fontsize=9)
ax.set(xticks=[0,1],xticklabels=['6 views','12 views'],ylim=(0,2.65),ylabel='Vehicle center error (m)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(fontsize=8,loc='upper right');ax.set_title('(d) Continuous localization recovery',loc='left',weight='bold',fontsize=11)
ax=fig.add_subplot(g[1,2]);ax.axis('off');ax.text(0,1,'What is supported',va='top',weight='bold',fontsize=13,color='#173c61')
ax.text(0,.87,'Original: confidence 0.70 / 0.62,\nbut center error 2.07 / 1.95 m.\nIt is not a disappearing detection.\n\nLocal correction: 0.086 / 0.030 m;\nvehicle-box IoU: 0.35 / 0.37\n                         → 0.84 / 0.80.',va='top',fontsize=11,linespacing=1.3)
ax.text(0,.27,'Ordinary deletion also recovers.\nOne exposed train frame; no independent\nscene, planning or safety confirmation.\nNo proof that generative repair is needed.',va='top',fontsize=10,color='#8b4d34',weight='bold',linespacing=1.25)
fig.suptitle('Biased reconstructed surfaces shift a confident vehicle detection by about 2 metres',x=.035,ha='left',fontsize=17,weight='bold')
fig.text(.04,.093,'scene-0004 · VGGT-Ω original 512 checkpoint · BUILD global scale already controlled · 22 target returns are late by about 1.6–1.7 m.',fontsize=10,color='#405a70')
fig.text(.04,.060,'A saved local mesh correction changes 23 rays (one outside the GT target); point intensity/order and nine real past sweeps are controlled.',fontsize=10,color='#405a70')
fig.text(.04,.028,'Oracle target/range information is extra. This supports a local geometry → sensor → perception effect within this adapter, not universal SOTA failure.',fontsize=10,color='#405a70')
for ext in ['png','pdf','svg']:fig.savefig(O/f'F18_Local_Mesh_To_Perception.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
