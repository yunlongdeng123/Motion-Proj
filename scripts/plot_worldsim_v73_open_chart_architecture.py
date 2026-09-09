"""绘制实际开放曲面组件；灰色视觉通路不代表r3进行了视觉训练。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig,ax=plt.subplots(figsize=(15,5.3))
ax.set(xlim=(0,15),ylim=(0,5)); ax.axis('off')
def box(x,y,w,h,label,color='#f4f5f6'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.12',
                              facecolor=color,edgecolor='#34383b',linewidth=1.5))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=11)
def arrow(a,b,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,
                                color='#404040',linewidth=1.5,linestyle='--' if dashed else '-'))
box(.2,2.15,1.55,1.35,'Build LiDAR\ncanonical points','#d9e7f5')
box(.2,.2,3.25,1.1,'Frozen visual prefix → trainable DPT\nJoint interface; inactive in LiDAR r3','#ededed')
box(2.25,2.05,2.3,1.55,'≤1024 evidence queries\n+512 completion queries\n3 spatial updates','#e0e9dc')
box(5.15,2.05,2.0,1.55,'64 chart anchors\nFPS on initial support\nupdated 3D centers','#fff0cb')
box(7.75,1.85,3.0,1.95,'Local PCA frame +\nlearned normal / UV height\n4×4 shared vertices per chart\nfixed metric span from Actor size','#dcebed')
box(11.4,2.05,3.0,1.55,'1024 vertices / 1152 triangles\nOpen between charts\nSame train / export surface','#efdfeb')
box(7.75,.2,6.65,1.0,'Observed point→surface + finite-beam free + box envelope\nLiteral first-hit / early / missing evaluation; no opacity','#f5f5f5')
arrow((1.8,2.82),(2.18,2.82)); arrow((4.6,2.82),(5.08,2.82))
arrow((7.2,2.82),(7.68,2.82)); arrow((10.8,2.82),(11.32,2.82))
arrow((2.5,1.35),(3.3,1.98),True); arrow((12.9,1.99),(12.9,1.28))
ax.text(7.5,4.55,'Open local surface charts: support allocation without global closure',ha='center',fontsize=17)
ax.text(7.5,4.12,'Existing local heightfield idea; new support count, scale, anchors and shared chart decoder. Benefits remain untested.',ha='center',fontsize=10)
fig.subplots_adjust(left=.02,right=.99,bottom=.03,top=.98)
out=Path(__file__).resolve().parents[1]/'docs/autoresearch/worldsim_v73/open_charts'
out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf']: fig.savefig(out/('V73_OPEN_CHART_ARCHITECTURE.'+ext),dpi=150)
print('open-chart component diagram saved')
