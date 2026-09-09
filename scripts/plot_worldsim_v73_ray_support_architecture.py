"""绘制本轮实际监督通路；不把LiDAR控制画成已训练视觉基座。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

fig,ax=plt.subplots(figsize=(15,5.8)); ax.set(xlim=(0,15),ylim=(0,5.5)); ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.12',facecolor=color,edgecolor='#34383b',linewidth=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#404040',linewidth=1.5))
box(.2,2.8,1.65,1.15,'Build LiDAR\ncanonical points','#d9e7f5')
box(2.35,2.55,2.45,1.65,'Spatial Query decoder\n64 open UV charts\nSame as LiDAR r3','#e0e9dc')
box(5.3,2.65,2.25,1.45,'Explicit surface\n1024 vertices\n1152 triangles','#efdfeb')
box(8.1,2.4,3.1,1.95,'Ray-conditioned closest point\nfull anisotropic metric\n64 rays × 512 faces per block\nNo opacity or unknown labels','#fff0cb')
box(11.85,2.65,2.7,1.45,'Surface attraction\nlateral + along-ray distance\nPosition / normal / height\nreceive gradients','#dcebed')
box(5.3,.3,3.25,1.1,'Owned original first returns\nFIT full-track labels only\norigin + direction + range','#d9e7f5')
box(9.3,.3,5.25,1.1,'Keep point→surface + finite-beam free + box\nEvaluate literal hit / early / missing on the same mesh\nAttraction does not guarantee correct first hit','#f4f5f6')
for a,b in [((1.9,3.37),(2.28,3.37)),((4.85,3.37),(5.23,3.37)),((7.6,3.37),(8.03,3.37)),((11.25,3.37),(11.78,3.37)),((8.15,1.45),(9.15,2.33)),((13.2,2.59),(13.2,1.46))]: arrow(a,b)
ax.text(7.5,5.02,'Ray support r4: one supervision change on the r3 open surface',ha='center',fontsize=17)
ax.text(7.5,4.56,'LiDAR control: no visual prefix / DPT loaded. Attraction is distinct from literal first-hit evaluation.',ha='center',fontsize=11)
fig.subplots_adjust(left=.02,right=.99,bottom=.03,top=.98)
out=Path(__file__).resolve().parents[1]/'docs/autoresearch/worldsim_v73/ray_support'; out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']: fig.savefig(out/('V73_RAY_SUPPORT_ARCHITECTURE.'+ext),dpi=150)
