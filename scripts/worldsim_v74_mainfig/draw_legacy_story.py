import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle,FancyArrowPatch,FancyBboxPatch
from PIL import Image
B=pathlib.Path(__file__).resolve().parent;OLD=B.parent/'v74_return';OUT=B.parent.parent/'outputs/V74_Main_Paper_Figures';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
red='#dc3545';teal='#008577';ink='#172b46';gray='#8794a5';gold='#ffc400'
a=np.load(next((OLD/'sources').rglob('VGGT-native.npz')));ctx=np.load(B/'legacy_context/VGGT-native.npz');meta=json.loads((B/'legacy_context/VGGT-native.json').read_text())
im=Image.open(B/'legacy_context/VGGT-native.jpg');nw,nh=ctx['network_hw'][::-1];iw,ih=im.size
K=ctx['K'];T=ctx['camera_from_actor'];V=a['vertices'];tri=V[a['faces']];early=np.isfinite(a['first'])&(a['first']<a['ranges']-.2);ids=np.unique(a['face_ids'][early]);ids=ids[ids>=0]
rid=int(a['selected_ray']);o=a['origins'][rid].astype(float);d=a['directions'][rid].astype(float);r=float(a['ranges'][rid]);t=float(np.load(OLD/'audit/VGGT-native_recast.npz')['first'][rid]);p=o+t*d;g=o+r*d;gap=r-t
def project(x):
 z=x@T[:3,:3].T+T[:3,3];uv=z@K.T;uv=uv[...,:2]/uv[...,2:3];return uv*[iw/nw,ih/nh]
polys=project(tri[ids].reshape(-1,3)).reshape(-1,3,2);uvp=project(p[None])[0];uvg=project(g[None])[0];box=np.array(meta['bbox_from_build_points'])*[iw/nw,ih/nh,iw/nw,ih/nh]
fig=plt.figure(figsize=(12,12.5),facecolor='white');gs=fig.add_gridspec(3,2,height_ratios=[1.1,1.25,.84],hspace=.39,wspace=.24,left=.105,right=.97,top=.90,bottom=.065)
fig.text(.07,.967,'A complete-looking car can return LiDAR too early',fontsize=20,fontweight='bold',color=ink)
fig.text(.07,.938,'Measured legacy case  |  scene-0520  |  adapted VGGT + LiDAR fusion surface',fontsize=11,color=gray)
ax=fig.add_subplot(gs[0,0]);ax.imshow(im);ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,ec=gold,lw=2));ax.add_collection(PolyCollection(polys,facecolors=red,edgecolors='none',alpha=.85));ax.set_title('(a) Locate the white car in RGB',loc='left',fontweight='bold',color=ink);ax.axis('off');ax.text(.02,.02,'Yellow: target car   Red: early-hit triangles',transform=ax.transAxes,color='white',fontsize=10,bbox=dict(fc=ink,alpha=.85,ec='none',pad=4))
ax=fig.add_subplot(gs[0,1]);ax.imshow(im);ax.add_collection(PolyCollection(polys,facecolors=red,edgecolors=red,lw=.4,alpha=.72));ax.scatter(*uvp,s=85,c=red,edgecolors='white',zorder=5);ax.set_xlim(box[0]-35,box[2]+35);ax.set_ylim(box[3]+30,box[1]-45);ax.set_title('Actual projected prediction, enlarged',loc='left',fontweight='bold',color=ink);ax.axis('off');ax.annotate('Example early intersection',xy=uvp,xytext=(box[0]+20,box[1]-12),color=red,fontsize=10,fontweight='bold',arrowprops=dict(arrowstyle='->',color=red,lw=1.5),bbox=dict(fc='white',ec='none',alpha=.9,pad=3))
ax=fig.add_subplot(gs[1,0]);target=a['target_points'];build=a['build_points'];u=d[:2]/np.linalg.norm(d[:2]);cross=np.array([-u[1],u[0]])
def bev(x):return (x[...,:2]-o[:2])@np.stack([u,cross],axis=1)
bb=bev(build);qq=bev(target);bp=bev(p);bg=bev(g);ax.scatter(bb[:,0],bb[:,1],s=2,c='#ccd7dd',label='BUILD support');ax.scatter(qq[:,0],qq[:,1],s=6,c=teal,label='Real heldout returns');ax.add_collection(PolyCollection(bev(tri[ids]),facecolors=red,alpha=.55,edgecolors='none'));sx,sy,sz=a['size'];corners=np.array([[-sx/2,-sy/2],[sx/2,-sy/2],[sx/2,sy/2],[-sx/2,sy/2],[-sx/2,-sy/2]]);cb=bev(corners);ax.plot(cb[:,0],cb[:,1],ls='--',c=gray,lw=1);ax.scatter(0,0,marker='D',s=90,c=ink);ax.text(.1,.4,'LiDAR\nsensor',va='center',fontsize=10,color=ink);ax.plot([0,bg[0]],[0,bg[1]],color=teal,lw=2,alpha=.9);ax.plot([0,bp[0]],[0,bp[1]],color=red,lw=1.5,ls='--');ax.scatter(*bp,s=80,c=red,zorder=5);ax.scatter(*bg,s=60,c=teal,marker='x',lw=2,zorder=6);ax.set_aspect('equal');ax.set_xlim(-.6,10);ax.set_ylim(min(-3.8,cb[:,1].min()-.35),max(2.5,cb[:,1].max()+.35));ax.set_xlabel('Ground distance along beam direction [m]');ax.set_ylabel('Ground transverse distance [m]');ax.set_title('(b) Same heldout beam in BEV',loc='left',fontweight='bold',color=ink);ax.legend(loc='lower left',fontsize=9,frameon=False)
ax=fig.add_subplot(gs[1,1]);ax.axvspan(t,r,color=red,alpha=.1);ax.axhline(0,color='#e1e6ec',lw=1);ax.plot([r-1,t],[0,0],color=red,lw=3);ax.plot([t,r],[0,0],color=teal,lw=2,ls='--');ax.scatter(t,0,s=170,c=red,marker='s',zorder=5);ax.scatter(r,0,s=170,c=teal,marker='X',zorder=5);ax.annotate('',xy=(t,.30),xytext=(r,.30),arrowprops=dict(arrowstyle='<->',color=red,lw=2));ax.text((r+t)/2,.39,f'{gap:.3f} m early',ha='center',color=red,fontsize=18,fontweight='bold');ax.text(t,-.14,f'Predicted first hit\n{t:.3f} m',ha='center',va='top',color=red,fontsize=11);ax.text(r,-.14,f'Real first return\n{r:.3f} m',ha='center',va='top',color=teal,fontsize=11);ax.text((r+t)/2,.12,'Measured free space',ha='center',color=red,fontsize=10);ax.set_xlim(t-.30,r+.3);ax.set_ylim(-.42,.60);ax.set_yticks([]);ax.set_xlabel('Range along the same beam [m]');ax.set_title('Range zoom — drawn to scale',loc='left',fontweight='bold',color=ink);ax.spines['left'].set_visible(False);ax.text(.0,-.29,'77 / 752 heldout rays are early by > 0.20 m.\nBoth heldout times: 59 / 557 and 18 / 195.',transform=ax.transAxes,fontsize=10,color=ink,linespacing=1.6)
ax=fig.add_subplot(gs[2,0]);start=t-.40;end=r+.34
for y,x,c in [(1,r,teal),(0,t,red)]:
 ax.add_patch(Rectangle((start,y-.22),x-start,.44,fc='#cdece5',ec='white'))
 ax.add_patch(Rectangle((x,y-.22),.05,.44,fc=c,ec='white'))
 ax.add_patch(Rectangle((x+.05,y-.22),end-x-.05,.44,fc='#dfe4ec',ec='white',hatch='///'))
 ax.text(start+.03,y,'Free',va='center',fontsize=10,color=teal)
 ax.text(end-.02,y,'Unknown',va='center',ha='right',fontsize=9,color='#606c7c')
ax.plot([t,t],[-.34,1.34],ls=':',color=red,lw=1);ax.plot([r,r],[-.34,1.34],ls=':',color=teal,lw=1);ax.set_yticks([0,1],['Pred LiDAR','GT LiDAR']);ax.set_xlim(start,end);ax.set_ylim(-.6,1.5);ax.set_xlabel('Range [m]; 5 cm occupied endpoint cells');ax.set_title('(c) Sensor-to-occupancy consequence',loc='left',fontweight='bold',color=ink);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.text(0,-.3,'Single-beam inverse sensor model. Cells behind a hit stay UNKNOWN.',transform=ax.transAxes,fontsize=9,color=gray)
ax=fig.add_subplot(gs[2,1]);ax.axis('off')
for y,text,color in [(.83,'Phantom enters measured free space',red),(.54,'Simulated LiDAR returns too early',red),(.25,'Occupied endpoint moves toward sensor',ink)]:
 ax.add_patch(FancyBboxPatch((.01,y-.10),.96,.20,boxstyle='round,pad=.015',transform=ax.transAxes,fc='#fff4f4' if color==red else '#eef3f8',ec='none'));ax.text(.49,y,text,ha='center',va='center',transform=ax.transAxes,color=color,fontweight='bold',fontsize=11)
 if y>.3:ax.annotate('',xy=(.49,y-.19),xytext=(.49,y-.11),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color=gray,lw=1.5))
ax.text(.02,-.04,'Planning / collision risk: requires closed-loop evaluation.\nEarly return alone does not establish false-safe.',transform=ax.transAxes,fontsize=10,color=gray,linespacing=1.6)
for ext in ['png','pdf','svg']:fig.savefig(F/('fig01_legacy_white_car_story.'+ext),dpi=220,facecolor='white')
plt.close(fig)
e=OUT/'evidence';e.mkdir(exist_ok=True);(e/'legacy_story.json').write_text(json.dumps({'source_case':str(next((OLD/'sources').rglob('VGGT-native.npz'))),'selected_ray':rid,'range_m':r,'first_m':t,'gap_m':gap,'early_count':int(early.sum()),'rays':len(early),'red_faces':ids.tolist(),'rgb':'BUILD camera frame, actor-frame projection; not simultaneous QUERY photography','occupancy':'demonstrated single-beam inverse sensor model; no closed-loop rollout','false_safe':'NOT_ESTABLISHED'},indent=2))
print(F/'fig01_legacy_white_car_story.png')
