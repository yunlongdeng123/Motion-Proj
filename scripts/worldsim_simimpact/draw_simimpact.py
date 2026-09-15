"""以保存的真实输入和原生运行结果绘制可追溯科研图。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,FancyBboxPatch,FancyArrowPatch
from PIL import Image
W=Path(__file__).resolve().parent;B=W/'visual_bundle';O=W.parents[1]/'outputs/Simulation_Impact_Research';O.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
RED='#C74343';BLUE='#276CA1';GREEN='#27846B';DARK='#26364A'
rows=json.loads((B/'paired_collision_geometry.json').read_text())
fig=plt.figure(figsize=(14,10),layout='constrained');grid=fig.add_gridspec(3,3,width_ratios=[1.45,1,.95])
for r,name in enumerate(['scene-0013','scene-0038','scene-0041']):
    g=np.load(B/name/'geometry.npz');a=json.loads((B/name/'audit.json').read_text());term=a['native_terminal'];xy=g['initial_rgb_uv'];z=g['initial_rgb_camera_z']
    ax=fig.add_subplot(grid[r,0]);ax.imshow(Image.open(B/name/'raw_initial.jpg'))
    valid=(z>0)&(xy[:,0]>=0)&(xy[:,0]<800)&(xy[:,1]>=0)&(xy[:,1]<450)
    ax.scatter(xy[valid,0],xy[valid,1],s=6,c=RED,alpha=.65,linewidths=0)
    if valid.any():
        left,bottom=np.min(xy[valid],axis=0);right,top=np.max(xy[valid],axis=0)
        ax.add_patch(Polygon([[left-8,bottom-8],[right+8,bottom-8],[right+8,top+8],[left-8,top+8]],fill=False,edgecolor='#FFD44C',lw=2))
    ax.set_title(f'{name}  |  (a) Real RGB at start',loc='left',weight='bold',color=DARK);ax.axis('off')
    ax.text(.01,.02,'Red: GS centers later counted as collision',transform=ax.transAxes,color='white',fontsize=9,bbox={'facecolor':'#17202B','alpha':.8,'pad':4,'edgecolor':'none'})
    ax=fig.add_subplot(grid[r,1]);p=g['background'];l=g['lidar'];v=g['verts'];c=g['collision_points'];center=v.mean(0)
    # Show the local collision region and measured sparse returns, not a fabricated solid object.
    ax.scatter(p[::3,2],-p[::3,0],s=.5,c='#AEB7BF',alpha=.3,rasterized=True,label='GS centers')
    ax.scatter(l[:,2],-l[:,0],s=1.3,c=BLUE,alpha=.5,rasterized=True,label='Raw LiDAR (provisional)')
    ax.scatter(c[:,2],-c[:,0],s=7,c=RED,alpha=.85,label='Counted centers')
    outline=v[[0,1,4,5,0]];ax.plot(outline[:,2],-outline[:,0],c=DARK,lw=1.7,label='Ego footprint')
    path=g['path'];ax.plot(path[:,2],-path[:,0],c=GREEN,lw=1.5)
    ax.set_xlim(center[2]-4,center[2]+4);ax.set_ylim(-center[0]-3.2,-center[0]+3.2);ax.set_aspect('equal')
    ax.set_title('(b) Terminal BEV',loc='left',weight='bold',color=DARK);ax.set_xlabel('Forward in scene (m)');ax.set_ylabel('Left in scene (m)')
    if r==0:ax.legend(fontsize=7,loc='upper right',markerscale=2)
    ax=fig.add_subplot(grid[r,2]);subset=[next(x for x in rows if x['scene']==name and x['geometry']==k) for k in ['identity','native_stride2','native_stride4']]
    vals=[x['counts'][-1] for x in subset];ax.bar([0,1,2],vals,color=[RED,BLUE,GREEN],width=.58)
    ax.axhline(100,c=DARK,ls='--',lw=1);ax.text(2.45,104,'Threshold: >100',ha='right',fontsize=8)
    for i,val in enumerate(vals):ax.text(i,val+5,str(val),ha='center',weight='bold')
    ax.set_xticks([0,1,2],['All centers','Every 2nd','Every 4th']);ax.tick_params(axis='x',labelsize=8)
    ax.set_ylim(0,275);ax.set_ylabel('Centers inside the same ego volume');ax.set_title('(c) Same-pose collision readout',loc='left',weight='bold',color=DARK)
    ax.text(.02,.96,f'Native stop: {term["simulated_seconds"]:.2f} s\nRoute completion: {term["route_completion"]:.1%}\nAll counted labels: terrain',transform=ax.transAxes,va='top',fontsize=9)
fig.suptitle('The simulator collision readout depends on point sampling',fontsize=17,weight='bold',color=DARK)
fig.supxlabel('Published HUGSIM + LTF, 3 frozen scenes. Density changes affect collision readout at a fixed pose.\nRGB/LiDAR overlays use the published global transform; exact correspondence is pending. Physical collision truth is not established.',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F01_Native_Collision_Evidence.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)

fig,ax=plt.subplots(figsize=(14,4.5));ax.set_xlim(0,14);ax.set_ylim(0,4.5);ax.axis('off')
def box(x,y,w,h,text,color='#EAF0F5'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.08',fc=color,ec='#8090A1',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color=DARK)
def arrow(a,b,label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color='#566B80',lw=1.4))
    if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.15,label,ha='center',fontsize=9,color=DARK)
box(.15,2.45,2.15,1,'Real driving log\nRGB + poses');box(3.05,2.45,2.15,1,'Published HUGSIM\nscene geometry');box(5.95,2.45,2.15,1,'Native renderer\nRGB');box(8.85,2.45,2.15,1,'Frozen LTF\nplanned trajectory');box(11.75,2.45,2.05,1,'iLQR + bicycle\nexecuted ego pose')
for x in [2.3,5.2,8.1,11.]:arrow((x,2.95),(x+.75,2.95))
box(5.95,.6,2.15,1,'Selected GS centers\nsemantic + opacity','#F8E4E0');box(8.85,.6,2.15,1,'Inside ego volume\ncount > 100?','#F8E4E0');box(11.75,.6,2.05,1,'Collision flag\nterminate / continue','#F8E4E0')
arrow((4.13,2.45),(5.95,1.1));arrow((8.1,1.1),(8.85,1.1));arrow((11.,1.1),(11.75,1.1));arrow((12.77,2.45),(9.93,1.6),'same pose')
box(.15,.6,4.9,1,'Diagnostic: official feed-forward depth\nVGGT / Ω / DVGT-1 / Pi3X\nCalibration + visible support + scale','#E1F0E9');arrow((5.05,1.1),(5.95,1.1))
ax.text(7,4.12,'Geometry must reach a consumed sensor channel or a physical decision',ha='center',fontsize=16,weight='bold',color=DARK)
ax.text(7,.1,'This LTF configuration consumes no LiDAR first-return input. A depth-only change has no direct path to its policy.',ha='center',fontsize=10,color=DARK)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F00_Architecture.{ext}',dpi=200,bbox_inches='tight')
plt.close(fig)
print(O)

audit=json.loads((W/'paired_control_audit.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
for ax,entry in zip(axes,audit):
    controls=entry['fixed_iteration_controls'];values=[c['summary']['route_completion']*100 for c in controls]
    ax.bar(np.arange(3),values,color=[RED,BLUE,GREEN],width=.6)
    for j,c in enumerate(controls):
        s=c['summary'];status='Collision' if s['collision'] else ('Route complete' if s['route_completion']>=1 else 'Route departure')
        ax.text(j,values[j]+2,f'{status}\n{s["simulated_seconds"]:.2f} s',ha='center',fontsize=8)
    ax.set_title(entry['scene'],weight='bold',color=DARK);ax.set_ylim(0,126);ax.set_xticks([0,1,2],['All centers','Every 2nd','Every 4th']);ax.tick_params(axis='x',labelsize=9)
    ax.set_ylabel('Official route completion (%)');ax.axhline(100,c='#8796A4',ls=':',lw=1)
fig.suptitle('Collision sampling can change how the same closed-loop drive ends',fontsize=16,weight='bold',color=DARK)
fig.supxlabel('Fixed-iteration controller control: same RGB, policy and dynamics; common pose prefixes match exactly (0 m difference).\nRemoves the official 50 ms stopping cap. Sampling sensitivity is not evidence of a natural feed-forward geometry failure.',fontsize=9)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F02_Controlled_Closed_Loop.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
