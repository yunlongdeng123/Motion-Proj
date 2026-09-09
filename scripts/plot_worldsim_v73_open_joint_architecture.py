"""最终视觉重接候选组件图，明确冻结前缀、可训练几何与物理读出。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

fig,ax=plt.subplots(figsize=(15,6)); ax.set(xlim=(0,15),ylim=(0,6)); ax.axis('off')
def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.03,rounding_size=.10',
                              facecolor=color,edgecolor='#34383b',linewidth=1.3))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.5)
def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#454545',linewidth=1.4))
box(.2,3.8,1.65,1.05,'Build images\ncalibrated views','#d9e7f5')
box(2.35,3.8,2.1,1.05,'VGGT aggregator\nFrozen + cached','#e8e8e8')
box(4.95,3.55,2.1,1.55,'Native DPT\nTrainable\nMulti-scale features\n+ metric depth','#fbe7bb')
box(.2,1.7,2.0,1.15,'Sparse build LiDAR\nActor coordinates','#d9e7f5')
box(4.95,1.65,2.1,1.25,'Native + LiDAR seeds\nKnown transforms\nLocal feature reads','#dcebdc')
box(7.7,2.65,2.2,1.6,'Spatial Query decoder\n64 open UV charts\nTrainable position\nnormal + height','#dcebdc')
box(10.55,2.65,1.95,1.6,'Explicit surface\n1024 vertices\n1152 triangles','#ecdfec')
box(13.05,2.65,1.7,1.6,'Literal first hit\nEarly / miss\nFree intrusion\nSurface coverage','#dcebed')
box(7.7,.3,7.05,1.25,'FIT supervision on the same surface\nPoint-to-surface + finite-beam free + box\nActual first-depth error; nearest-surface attraction on missed rays','#fff0cb')
for a,b in [((1.9,4.32),(2.28,4.32)),((4.5,4.32),(4.88,4.32)),
            ((6.,3.5),(6.,2.97)),((2.25,2.27),(4.88,2.27)),((7.1,2.27),(8.,2.58)),
            ((7.1,4.32),(7.63,3.8)),((9.95,3.45),(10.48,3.45)),
            ((12.55,3.45),(12.98,3.45)),((11.5,2.58),(11.5,1.62))]: arrow(a,b)
ax.text(7.5,5.7,'Final visual reconnection: Joint r7 versus LiDAR r6',ha='center',fontsize=17)
ax.text(3.6,.82,'Joint also uses native build-depth supervision.\nLiDAR control omits the visual / native branch.\nWhole-system comparison, not feature-only attribution.',
        ha='center',va='center',fontsize=10)
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
out=Path(__file__).resolve().parents[1]/'docs/autoresearch/worldsim_v73/ray_support'
for ext in ['png','pdf']: fig.savefig(out/('V73_OPEN_JOINT_COMPONENTS.'+ext),dpi=160)
