"""原生传感器闭环组件图：明确已完成数据准备与待实测的闭环。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'svg.fonttype':'none','pdf.fonttype':42})
O=Path(__file__).resolve().parents[2]/'outputs/Simulation_Impact_Research'
fig,ax=plt.subplots(figsize=(14,6.6));ax.set(xlim=(0,14),ylim=(0,6.6));ax.axis('off')
fig.patch.set_facecolor('#fafbfe')
def box(x,y,w,h,title,body,color='#e9eff9'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=.09',facecolor=color,edgecolor='#8a98ad',linewidth=1))
    ax.text(x+w/2,y+h-.28,title,ha='center',va='top',fontsize=12,weight='bold',color='#13283f')
    ax.text(x+w/2,y+.2,body,ha='center',va='bottom',fontsize=10.5,color='#28445f',linespacing=1.5)
def arrow(a,b,label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,lw=1.5,color='#3c5674'))
    if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=9,color='#3c5674')
ax.text(.3,6.2,'Native RGB + LiDAR simulation: evidence path',fontsize=20,weight='bold',color='#13283f')
ax.text(.3,5.76,'Integration in progress  |  Scene fitting uses additional metric observations  |  No new closed-loop result yet',fontsize=11,color='#75521c')
box(.3,3.6,2.5,1.6,'Measured log','6 cameras + LiDAR\nCalibration + actor tracks','#e3f1eb')
box(3.45,3.6,2.9,1.6,'Official SplatAD','Fit native scene asset\n50% sensor observations')
box(7,3.6,2.7,1.6,'Novel-pose sensors','Rendered RGB + LiDAR\nExecuted ego pose + time')
box(10.4,3.6,3.15,1.6,'Official driving modules','TransFuser RGB + LiDAR\nController + vehicle dynamics')
arrow((2.8,4.4),(3.45,4.4));arrow((6.35,4.4),(7,4.4));arrow((9.7,4.4),(10.4,4.4))
ax.plot([12,12,8.3],[3.6,3.08,3.08],color='#3c5674',lw=1.5);arrow((8.3,3.08),(8.3,3.6))
ax.text(10.4,2.85,'Executed state updates the next sensor frame',fontsize=10,ha='center',color='#3c5674')
box(.3,.95,6.05,1.38,'Held-out sensor reference','Range error / early return / missed return / camera correspondence','#e3f1eb')
box(7,.95,6.55,1.38,'Geometry intervention and outcome','Natural feed-forward geometry errors + coverage / scale / recovery controls\nTrajectory, physical clearance and contact; retain negative cases','#fff1da')
arrow((1.55,3.6),(1.55,2.33));ax.text(1.8,2.95,'50% held out',fontsize=9,color='#3c5674')
arrow((8.3,2.8),(8.3,2.33))
ax.text(.3,.32,'Native scene fitting is a metric simulator reference, not an equal-input ranking or a newly trained feed-forward method.',fontsize=10,color='#566378')
fig.tight_layout(pad=.5)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F11_Native_Sensor_Architecture.{ext}',dpi=180,bbox_inches='tight')
