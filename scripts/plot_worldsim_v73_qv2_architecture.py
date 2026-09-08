"""Q-v2已实现模块图，明确训练路径与固定拓扑边界。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig,ax=plt.subplots(figsize=(15,6.7)); ax.set(xlim=(0,15),ylim=(-.8,6)); ax.axis('off')
def box(x,y,w,h,text,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#333333',lw=1.6))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#333333',lw=1.6))
box(.2,4.05,1.55,1.05,'24 build\nimages','#fff1ca')
box(2.15,4.05,2.15,1.05,'Frozen VGGT\nfull-window prefix','#eeeeee')
box(4.75,4.05,2.05,1.05,'Trainable DPT\n4 scales + depth\nbuild-depth supervision','#d8eafc')
box(.2,1.9,1.55,1.1,'Build LiDAR\ncalibration\nactor poses','#e1f0df')
box(2.15,1.9,2.15,1.1,'Evidence queries\nLiDAR + native depth\n(no output patches)','#e1f0df')
box(4.75,.35,2.05,1.1,'Known actor size\n642 template vertices\n1,280 fixed faces','#eeeeee')
box(7.4,2.05,3.05,2.85,'3 refinement stages\n\nMesh-edge messages\n+ nearby evidence\n+ projected local attention\n\nUpdate shared vertices','#d8eafc')
box(11.0,3.8,3.55,1.25,'One explicit triangle mesh\nshared vertices / no opacity','#ffe2ca')
box(11.0,1.5,3.55,1.55,'Same surface in all objectives\ncoverage + beam free + envelope\n\nLiteral first-hit evaluation','#f4e2ef')
for a,b in [((1.85,4.58),(2.08,4.58)),((4.4,4.58),(4.67,4.58)),((6.9,4.58),(7.32,4.58)),
            ((1.85,2.45),(2.08,2.45)),((4.4,2.45),(7.32,2.45)),((6.9,.9),(8.1,1.98)),
            ((10.55,4.4),(10.92,4.4)),((12.77,3.72),(12.77,3.13)),((5.75,3.98),(4.05,3.08))]: arrow(a,b)
ax.text(7.5,5.66,'Q-v2: trainable geometry + shared-vertex surface generation + physical constraints',ha='center',fontsize=15)
ax.text(5.8,3.4,'Native depth seeds',ha='center',fontsize=9)
ax.text(7.5,-.45,'Fixed sphere topology is a prior; it does not guarantee no folds or correct hidden surfaces.\nNo unknown-space labels. Upper aggregation and new boundary / correspondence losses are unchanged.',fontsize=9,ha='center')
out=Path(__file__).resolve().parents[1]/'docs/autoresearch/worldsim_v73/qv2'
out.mkdir(parents=True,exist_ok=True)
for suffix in ['png','pdf','svg']:
    path=out/('V73_QV2_ARCHITECTURE.'+suffix)
    fig.savefig(path,dpi=160,bbox_inches='tight')
    if suffix=='svg': path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
plt.close(fig)
