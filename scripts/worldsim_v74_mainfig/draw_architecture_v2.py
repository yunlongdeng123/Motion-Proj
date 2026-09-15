import pathlib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
F=pathlib.Path(__file__).resolve().parents[2]/'outputs/V74_Main_Paper_Figures/figures';F.mkdir(parents=True,exist_ok=True);ink='#172b46';gray='#698095'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(14,6.3));ax.set_xlim(0,14);ax.set_ylim(0,6.3);ax.axis('off')
def box(x,y,w,t,fc='#eaf0f7',h=.75):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',fc=fc,ec='none'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10.5,color=ink)
def arrow(x,y,u,v):ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='->',lw=1.5,color=ink))
ax.text(.15,5.92,'Components, information budgets and output contracts',fontsize=17,fontweight='bold',color=ink)
box(.15,4.1,1.5,'BUILD RGB\n6 / 12 views');box(2.05,4.1,2.1,'VGGT / Omega\nDVGT-1/2 / Pi3X');box(4.6,4.1,1.6,'Native depth\nand point maps');box(6.65,4.1,1.8,'Declared grid\nsurface adapter');box(9.1,3.35,1.8,'Hard first-hit\nraycasting');box(11.4,3.35,2.2,'Single-beam\noccupancy diagnostic')
for x,u in [(1.65,2.05),(4.15,4.6),(6.2,6.65)]:arrow(x,4.475,u,4.475)
arrow(8.45,4.475,9.1,3.9);arrow(10.9,3.725,11.4,3.725)
box(.15,2.45,1.5,'BUILD LiDAR\n+ PCA normals',fc='#fff1d7');box(2.05,2.45,2.1,'NKSR / NoKSR\nofficial pretrained');box(4.6,2.45,1.6,'Native learned\nfield / mesh')
arrow(1.65,2.825,2.05,2.825);arrow(4.15,2.825,4.6,2.825);arrow(6.2,2.825,9.1,3.6)
ax.text(6.6,2.6,'Extra LiDAR input',fontsize=10,color=gray)
box(6.55,5.03,2.0,'Known K / poses\n+ BUILD scale',fc='#fff1d7',h=.55);arrow(7.55,5.03,7.55,4.85)
box(9.1,1.85,1.8,'QUERY LiDAR\nreference only',fc='#e3f2ec');arrow(10,2.6,10,3.35)
box(.15,.65,1.5,'BUILD RGB');box(2.05,.65,2.1,'DGGT');box(4.6,.65,1.6,'Native\nGaussians');box(6.65,.65,1.8,'Gaussian\nrendering');box(9.1,.65,2.15,'RGB + opacity\n+ expected depth')
for x,u in [(1.65,2.05),(4.15,4.6),(6.2,6.65),(8.45,9.1)]:arrow(x,1.025,u,1.025)
ax.text(11.55,2.1,'Closed-loop planning\nnot evaluated',fontsize=11,color='#b74745')
ax.text(.15,.16,'No QUERY data enters reconstruction. Expected depth is not a LiDAR first return. Extra information is shown explicitly.',fontsize=10.5,color=gray)
fig.subplots_adjust(0,0,1,1)
for ext in ['png','pdf','svg']:fig.savefig(F/('architecture_components.'+ext),dpi=220,facecolor='white')
