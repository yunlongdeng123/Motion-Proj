import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from PIL import Image
B=pathlib.Path(__file__).resolve().parent;W=B/'white_witness';OUT=B.parents[1]/'outputs/V74_Main_Paper_Figures';F=OUT/'figures'
A=json.loads((W/'audit.json').read_text());P=np.load(W/'query_projection.npz');im=Image.open(W/'query_rgb.jpg');methods=['vggt','omega512','dvgt1','pi3x'];labels=['Official VGGT','VGGT-Omega 512*','DVGT-1','Pi3X'];colors=['#3e64bd','#9a57ac','#b86b13','#168475'];red='#d93649';teal='#008577';ink='#172b46';gray='#758395'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def project(x):
 c=x@P['camera_from_actor'][:3,:3].T+P['camera_from_actor'][:3,3];uv=c@P['K'].T;return uv[...,:2]/uv[...,2:3]
a=np.load(W/'vggt.npz');o=a['origin'];d=a['direction'];r=float(a['range']);gt=o+d*r;gtuv=project(gt[None])[0];points=P['gt_points'];bp=project(P['build_points']);bmin=np.maximum(bp.min(0),[0,0]);bmax=np.minimum(bp.max(0),im.size)
fig=plt.figure(figsize=(12.7,10.7),facecolor='white');gs=fig.add_gridspec(3,4,left=.13,right=.975,top=.87,bottom=.19,height_ratios=[1.25,.9,.72],hspace=.62,wspace=.34)
fig.text(.07,.964,'Four models return before the same measured low LiDAR return',fontsize=18,fontweight='bold',color=ink)
fig.text(.07,.931,f'One shared beam: scene-0520, ray {A["selected_ray"]}  |  12-image predictions  |  ordinary calibration + BUILD global scale',fontsize=10.5,color=gray)
ax=fig.add_subplot(gs[0,:2]);ax.imshow(im);ax.add_patch(Rectangle(bmin,bmax[0]-bmin[0],bmax[1]-bmin[1],fill=False,ec='#ffc400',lw=1.8));ax.add_patch(Rectangle(gtuv-[95,60],190,110,fill=False,ec=red,lw=1.7));ax.scatter(*gtuv,c=teal,marker='x',s=35);ax.axis('off');ax.set_title('(a) RGB context and the white car',loc='left',fontweight='bold',color=ink);ax.text(.02,.02,'Heldout RGB: visualization only',transform=ax.transAxes,color='white',fontsize=9,bbox=dict(fc=ink,ec='none',alpha=.8,pad=3))
ax=fig.add_subplot(gs[0,2:]);ax.imshow(im)
for i,(m,label,col) in enumerate(zip(methods,labels,colors)):
 x=np.load(W/(m+'.npz'));t=float(x['first']);p=o+t*d;uv=project(p[None])[0];triuv=project(x['triangle']);ax.add_collection(PolyCollection([triuv],facecolors=red,edgecolors=red,linewidths=1,alpha=.75));ax.scatter(*uv,s=55,c=red,edgecolors='white',lw=.6,zorder=5)
ax.scatter(*gtuv,s=75,c=teal,marker='x',lw=2.4,zorder=6);ax.set_xlim(max(0,gtuv[0]-195),min(im.width,gtuv[0]+195));ax.set_ylim(min(im.height,gtuv[1]+100),max(0,gtuv[1]-155));ax.axis('off');ax.set_title('Same image location, different 3D ranges',loc='left',fontweight='bold',color=ink);ax.annotate('Predicted intersections\nand real return overlap in RGB',xy=gtuv,xytext=(gtuv[0]-150,gtuv[1]+75),color=ink,fontsize=9,arrowprops=dict(arrowstyle='->',color=ink),bbox=dict(fc='white',alpha=.9,ec='none',pad=3))
ax=fig.add_subplot(gs[1,:2]);near=(np.linalg.norm(points[:,:2]-gt[:2],axis=1)<1.4);u=d[:2]/np.linalg.norm(d[:2]);s=(points-o)[:,:2]@u;z=points[:,2]+P['size'][2]/2
ax.scatter(s[near],z[near],s=5,color='#b6c9c5',label='Nearby measured returns');start=r-1.0;qs=np.array([o+d*start,o+d*(r+.14)]);xp=(qs-o)[:,:2]@u;zp=qs[:,2]+P['size'][2]/2;ax.plot(xp,zp,ls='--',lw=1.2,c=ink);ax.scatter(r*np.linalg.norm(d[:2]),gt[2]+P['size'][2]/2,c=teal,marker='X',s=90,zorder=8,label='GT first return')
for m,col in zip(methods,colors):
 x=np.load(W/(m+'.npz'));p=o+d*float(x['first']);ax.scatter(float(x['first'])*np.linalg.norm(d[:2]),p[2]+P['size'][2]/2,c=col,s=55,zorder=7)
ax.axhline(0,c=gray,ls=':',lw=1);ax.text((r-.9)*np.linalg.norm(d[:2]),.035,'Annotated box bottom (context only)',fontsize=8.5,color=gray);ax.set_xlim((r-1)*np.linalg.norm(d[:2]),(r+.2)*np.linalg.norm(d[:2]));ax.set_ylim(-.14,.48);ax.set_xlabel('Ground distance along beam direction [m]');ax.set_ylabel('Height relative to box bottom [m]');ax.set_title('(b) Side view of the same measured beam',loc='left',fontweight='bold',color=ink,fontsize=10.5);ax.grid(alpha=.15)
ax=fig.add_subplot(gs[1,2:]);ax.axvline(r,color=teal,lw=2,ls='--');
for i,(row,label,col) in enumerate(zip(A['rows'],labels,colors)):
 t=row['first_m'];y=3-i;ax.plot([t,r],[y,y],c=red,lw=2);ax.scatter(t,y,c=col,s=65,zorder=5);ax.scatter(r,y,c=teal,marker='x',s=45);ax.text(t-.03,y+.22,f'{row["gap_m"]:.3f} m',ha='center',color=col,fontsize=9)
ax.set_xlim(r-.85,r+.10);ax.set_ylim(-.65,3.65);ax.set_yticks(range(4),labels[::-1],fontsize=8.5);ax.set_xlabel('Range [m]');ax.set_title('Observed first range: '+f'{r:.3f} m',loc='left',fontweight='bold',color=ink,fontsize=10.5);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
ax=fig.add_subplot(gs[2,:]);ax.set_xlim(r-.9,r+.25);ax.set_ylim(-.65,4.65)
for y,t,label,col in [(4,r,'GT',teal)]+[(3-i,row['first_m'],label,col) for i,(row,label,col) in enumerate(zip(A['rows'],labels,colors))]:
 start=r-.9;end=r+.25;ax.add_patch(Rectangle((start,y-.30),t-start,.60,fc='#ccebe3',ec='white'));ax.add_patch(Rectangle((t,y-.30),.05,.60,fc=col,ec='white'));ax.add_patch(Rectangle((t+.05,y-.30),end-t-.05,.60,fc='#e2e6ed',ec='white',hatch='///'))
ax.set_yticks([4,3,2,1,0],['GT']+labels,fontsize=9);ax.tick_params(axis='y',length=0);ax.set_xlabel('Range [m]');ax.spines['left'].set_visible(False);ax.set_title('(c) An opaque first-hit sensor model puts occupied endpoints too close',loc='left',fontweight='bold',color=ink)
fig.text(.07,.041,'Red: predicted early intersection. Teal ×: measured return. This low return is below the annotated car box, so it is NOT labeled as confirmed car body.\nTriangles have max edges 0.020–0.115 m; all their vertices precede the GT return by >0.40 m along the beam. The error survives 50% confidence retention.\nOccupancy shown is a single-beam diagnostic: green = free, solid = occupied, hatched = unknown. No false-safe or closed-loop result is claimed.',fontsize=9.5,color=gray,linespacing=1.55)
for ext in ['png','pdf','svg']:fig.savefig(F/('fig06_shared_low_return_witness.'+ext),dpi=230)
print(F/'fig06_shared_low_return_witness.png')
