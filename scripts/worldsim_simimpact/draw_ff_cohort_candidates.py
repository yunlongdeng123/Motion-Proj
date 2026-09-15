"""真实道路上下文与全局 BEV，避免把路侧行人 IoU 下降画成前方障碍危害。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
from PIL import Image
from scipy.spatial.transform import Rotation
W=Path(__file__).resolve().parent;E=W/'evidence_ff_cohort';D=W/'evidence_ff_cohort_support';O=W.parents[1]/'outputs/Simulation_Impact_Research'
support=json.loads((D/'support.json').read_text());objects=json.loads((E/'objects.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(16,8.8));grid=fig.add_gridspec(2,3,left=.05,right=.97,top=.90,bottom=.13,hspace=.30,wspace=.30,height_ratios=[1.15,1])
ax=fig.add_subplot(grid[0,:2]);ax.imshow(Image.open(D/(support[0]['prefix']+'_rgb.jpg')));colors=['#ffd030','#4cd4de']
for s,c,label in zip(support,colors,['A: Pi3X candidate','B: DVGT-1 candidate']):
    x0,y0,x1,y1=s['image_box_xyxy'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec=c,lw=2));ax.annotate(label,xy=((x0+x1)/2,y0),xytext=(x0-210,y0-80 if label[0]=='A' else y1+80),color=c,fontsize=9,weight='bold',arrowprops={'arrowstyle':'->','color':c},bbox={'facecolor':'#192731','alpha':.8,'pad':3,'edgecolor':'none'})
ax.axis('off');ax.set_title('(a) Real scene-0028 · CAM_FRONT_RIGHT',loc='left',weight='bold',fontsize=12)
ax=fig.add_subplot(grid[0,2]);ax.add_patch(Rectangle((-1,-2.4),2,4.8,facecolor='#334f62'));ax.annotate('',xy=(0,7),xytext=(0,2.4),arrowprops={'arrowstyle':'->','lw':2,'color':'#334f62'});ax.text(1.5,0,'Ego',va='center',fontsize=10)
for s,c,label in zip(support,colors,['A','B']):
    r=next(r for r in objects if r['condition']=='real' and r['instance']==s['instance'] and r['scene']==s['scene']);x,y=r['center_ego'][:2];ax.scatter(y,x,s=65,edgecolors='#263f50',facecolors=c);ax.text(y-.9,x+.6,label,weight='bold');ax.plot([0,y],[0,x],color='#9cadb9',lw=.8,ls=':')
ax.set(xlim=(5,-19),ylim=(-3,25),xlabel='Left (+) / right (−), metres',ylabel='Forward (m)');ax.set_aspect('equal');ax.grid(alpha=.18);ax.set_title('(b) Both targets are far to the side',loc='left',weight='bold',fontsize=11)
def corners(g,T):
    w,l,h=g['wlh'];q=np.array(g['rotation']);rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();p=np.array([[l/2,w/2,0],[l/2,-w/2,0],[-l/2,-w/2,0],[-l/2,w/2,0]])@rot.T+g['center_lidar'];return (p@T[:3,:3].T+T[:3,3])[:,:2]
for j,s in enumerate(support):
    ax=fig.add_subplot(grid[1,j]);z=np.load(D/(s['prefix']+'_points.npz'));T=np.linalg.inv(z['lidar_from_ego']);xy=corners(s['gt'],T);ax.add_patch(Polygon(xy,fill=False,ec='#263d4e',lw=1.8,ls='--',label='GT box'));polys=[xy]
    for suffix,col,label in [('real','#3d9076','Real-input box'),(s['method']+'_six_fill_missing','#c8664d','6-view input box'),(s['method']+'_twelve_fill_missing','#8377b1','12-view input box')]:
        r=next(r for r in objects if r['condition']==suffix and r['instance']==s['instance'] and r['scene']==s['scene']);g=r['scores']['0.5']['nearest_same_class'];xy=corners(g,T);polys.append(xy);ax.add_patch(Polygon(xy,fill=False,ec=col,lw=1.4,label=label))
    pts=np.concatenate(polys);ax.set(xlim=(pts[:,0].min()-.25,pts[:,0].max()+.25),ylim=(pts[:,1].min()-.25,pts[:,1].max()+.25),xlabel='Forward (m)',ylabel='Left (m)');ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1))
    ax.set_title(f'({chr(99+j)}) '+('A · Pi3X' if j==0 else 'B · DVGT-1')+' · actual box geometry',loc='left',weight='bold',fontsize=11)
ax=fig.add_subplot(grid[1,2]);ax.axis('off');ax.text(0,1,'What changed?',weight='bold',fontsize=13,va='top',color='#263f51');ax.text(0,.87,'A: center error 0.18 → 0.40 / 0.48 m\n    BEV IoU 0.52 → 0.15 / 0.10\nB: center error 0.03 → 0.34 / 0.36 m\n    BEV IoU 0.69 → 0.19 / 0.18\n\nPredicted confidence stays 0.74–0.81.\nBoth are still detected. GT support:\n11 / 8 returns; middle-height: 8 / 7.\n\nNo local asset intervention here.\nNo demonstrated driving consequence.\nThese are localization candidates,\nnot collision / false-safe evidence.',va='top',fontsize=10.5,linespacing=1.28,color='#354b5e')
fig.suptitle('A box-overlap loss can be visible without being a serious driving failure',x=.04,ha='left',fontsize=17,weight='bold')
fig.text(.05,.074,'Two candidates frozen by downstream losses across both view budgets. Both are about 15m to the right of ego; all original missing rays are restored.',fontsize=10,color='#435e70')
fig.text(.05,.036,'Panels (c,d) show nearest same-class boxes, not verified track identities. Dense pedestrian groups and small footprints make IoU sensitive.',fontsize=10,color='#83533c')
for ext in ['png','pdf','svg']:fig.savefig(O/f'F20_Candidate_Context_And_Limits.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)
p=O/'F20_Candidate_Context_And_Limits.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
