"""从归档证据生成论文向量图、失败切片与表格。"""
import json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.collections import LineCollection
ROOT=Path(os.environ.get('WORLDSIM_PAPER_STAGE',Path(__file__).resolve().parent))
PAPER=ROOT/'paper';FIG=PAPER/'figures'/'v73';FIG.mkdir(parents=True,exist_ok=True)
D=json.loads((ROOT/'forensics/summary.json').read_text());M=['AdaPoinTr','VGGT-native','LiDAR-R8','Open-r3','Attraction-r4','First-r6']
R7=json.loads((ROOT/'forensics_r7/summary.json').read_text()) if (ROOT/'forensics_r7/summary.json').exists() else None
if R7:
 D['statistics'].update(R7['statistics']);D['selected'].update(R7['selected'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(FIG/(name+'.'+ext),dpi=190,bbox_inches='tight',pad_inches=.05)
 plt.close(fig)
def architecture():
 fig,ax=plt.subplots(figsize=(12,3.4));ax.set(xlim=(0,12),ylim=(0,3.4));ax.axis('off')
 def box(x,y,w,h,title,sub,color):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.06',facecolor=color,edgecolor='#687b8c',linewidth=.8));ax.text(x+w/2,y+h*.69,title,ha='center',va='center',weight='bold',fontsize=10);ax.text(x+w/2,y+h*.28,sub,ha='center',va='center',fontsize=8.4)
 def arrow(a,b,text=None):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#415364',linewidth=1.1))
  if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,text,ha='center',fontsize=8)
 box(.12,1.95,2,1,'Sparse observations','LiDAR + camera frames','#e8eff8')
 box(.12,.25,2,1,'Known rigid poses','Calibration + identities','#f0f2f4')
 box(2.65,1.95,2.3,1,'Geometry decoding','VGGT DPT + spatial queries','#e5f0ed')
 box(2.65,.25,2.3,1,'Canonical witnesses','Measured points + pre-hit rays','#e8eff8')
 box(5.5,1.95,2.75,1,'Canonical surface charts','One shared metric surface per actor','#f7eadc')
 box(5.5,.25,2.75,1,'Witness-constrained update','Protect first hits; complete support','#f7eadc')
 box(9,1.95,2.8,1,'Reusable surface','Mesh / closest point / rigid placement','#e8f0ed')
 box(9,.25,2.8,1,'Literal first intersection','Test the same exported triangles','#edf0f4')
 arrow((2.2,2.45),(2.55,2.45));arrow((2.2,.75),(2.55,.75));arrow((1.1,1.95),(2.7,1.25));arrow((5,2.45),(5.4,2.45));arrow((5,.75),(5.4,.75));arrow((6.9,1.32),(6.9,1.86));arrow((8.35,2.45),(8.9,2.45));arrow((10.4,1.87),(10.4,1.32));arrow((9,.75),(8.35,.75))
 ax.text(6.8,.0,'Orange: surface formulation; orthogonal update is a checked prototype, not a trained r7 component.',ha='center',fontsize=8,color='#75502c')
 save(fig,'architecture')
def oracle():
 methods=M+(['Joint-r7'] if R7 else []);fig,ax=plt.subplots(figsize=(9,4.2));y=np.arange(len(methods));v=np.array([[D['statistics'][m]['equal_log'][k]*100 for k in ['hit','any_hit','near']] for m in methods])
 cols=['#356b93','#efb069','#b6d5cb'];labels=['Literal first hit','Additional: select a later correct hit (oracle)','Additional: nearby surface, no correct ray crossing']
 left=np.zeros(len(methods))
 for a,b,c in zip([v[:,0],v[:,1]-v[:,0],v[:,2]-v[:,1]],cols,labels):ax.barh(y,a,left=left,color=b,label=c,height=.64);left+=a
 for i in range(len(methods)):
  for z in [0,1,2]:ax.text(v[i,z]-(.9 if z==0 else -.7),i+.02,f'{v[i,z]:.1f}',fontsize=8.2,ha='right' if z==0 else 'left',va='center',color='white' if z==0 else '#1d3545')
 ax.set_yticks(y,methods);ax.invert_yaxis();ax.set(xlim=(0,101),xlabel='Owned returns (%) / actor-normalized, equal log mean');ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=8.4,ncol=1);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
 save(fig,'oracle_ladder')
def pruning():
 fig,axs=plt.subplots(1,3,figsize=(10.6,3.3));x=np.arange(3);methods=['AdaPoinTr','Attraction-r4','First-r6']
 for ax,m in zip(axs,methods):
  s=D['statistics'][m]['equal_log'];conditions=['Original','Build-component\npruning','Early-face\nremoval oracle'];prefix=['','build_component_','oracle_early_face_']
  for off,k,col in [(-.23,'hit','#356b93'),(0,'early','#c45454'),(.23,'miss','#b0b6bd')]:ax.bar(x+off,[s[p+k]*100 for p in prefix],width=.22,color=col,label=k)
  ax.set(ylim=(0,90),title=m);ax.set_xticks(x,conditions,fontsize=8);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
 axs[0].set_ylabel('Owned returns (%)');axs[-1].legend(frameon=False,fontsize=8);save(fig,'pruning_oracles')
def plane_slice(name,ax):
 a=np.load(ROOT/('forensics_r7/cases' if name=='Joint-r7' else 'forensics/cases')/(name+'.npz'));ri=int(a['selected_ray']);v=a['vertices'];f=a['faces'];o=a['origins'][ri];d=a['directions'][ri];r=a['ranges'][ri];y=o+d*r;fid=int(a['face_ids'][ri]);cid=a['components'][fid]
 up=np.array([0.,0.,1.]);up-=np.dot(up,d)*d;up/=np.linalg.norm(up);n=np.cross(d,up);tri=v[f]-y;h=tri@n;segments=[];color=[];width=[]
 for k,(t,dist) in enumerate(zip(tri,h)):
  pts=[]
  for i,j in [(0,1),(1,2),(2,0)]:
   if dist[i]*dist[j]<0:pts.append(t[i]+dist[i]/(dist[i]-dist[j])*(t[j]-t[i]))
  if len(pts)==2:
   seg=np.array([[p@d,p@up] for p in pts]);segments.append(seg);color.append('#c9353f' if k==fid else '#e9a459' if a['components'][k]==cid else '#aeb9c4');width.append(2.5 if k==fid else 1.4 if a['components'][k]==cid else .7)
 ax.add_collection(LineCollection(segments,colors=color,linewidths=width,zorder=1));gap=float(r-a['first'][ri]);ax.axhline(0,color='#233e50',lw=.8);ax.axvspan(-.2,.2,color='#d6e9dc',alpha=.65,zorder=0);ax.scatter([-gap,0],[0,0],c=['#c9353f','#128e49'],s=[45,45],zorder=4)
 ax.annotate('',xy=(-.02,-.14),xytext=(-gap,-.14),arrowprops={'arrowstyle':'<->','lw':.9,'color':'#b44545'});ax.text(-gap/2,-.23,f'{gap:.3f} m early',ha='center',fontsize=8,color='#a02e33')
 ax.set(xlim=(-max(1.1,gap+.4),.65),ylim=(-.5,.65),xlabel='Range offset (m)',ylabel='In-plane offset (m)');ax.set_aspect('equal',adjustable='box');ax.set_title(name+' | face q='+f"{a['q'][fid]:.3f}",fontsize=10,loc='left');ax.grid(alpha=.13);ax.tick_params(labelsize=8)
def slices():
 fig,axs=plt.subplots(2,3,figsize=(12,5.8))
 for m,a in zip(M,axs.flat):plane_slice(m,a)
 fig.text(.5,.99,'Exact intersection of original triangles with a plane containing the selected sensor ray',ha='center',fontsize=11)
 fig.text(.5,.01,'Red: first-hit triangle   Orange: same connected chart   Green dot: heldout return   Green band: ±0.2 m',ha='center',fontsize=9)
 fig.subplots_adjust(wspace=.35,hspace=.46,top=.9,bottom=.11);save(fig,'failure_slices')
def tables():
 out=PAPER/'tables'/'v73';out.mkdir(exist_ok=True)
 def write(name,cols,head,rows):
  lines=[r'\begin{tabular}{'+cols+'}',r'\toprule',head+' '+chr(92)*2,r'\midrule']+[row+' '+chr(92)*2 for row in rows]+[r'\bottomrule',r'\end{tabular}'];(out/(name+'.tex')).write_text('\n'.join(lines)+'\n')
 rows=[]
 for m in M+(['Joint-r7'] if R7 else []):
  s=D['statistics'][m]['equal_log'];rows.append(m.replace('VGGT-native','VGGT native')+' & '+' & '.join(f'{s[k]*100:.2f}' for k in ['hit','any_hit','near','early','early_unsupported_component']))
 write('oracle','lrrrrr',r'Method & First $\uparrow$ & Any $\uparrow$ & Near $\uparrow$ & Early $\downarrow$ & Unsupported early',rows)
 d=json.loads((ROOT/'evidence/first_surface_r6_analysis.json').read_text());rows=[]
 for k,label,factor in [('hit_rate','Hit (pp)',100),('early_rate','Early (pp)',100),('miss_rate','Missing (pp)',100),('free_intrusion_m','Free (m)',1),('surface_distance_m','Distance (m)',1),('surface_recall_02','Recall (pp)',100)]:
  a=d['paired_final_minus']['ray_support_r4']['development'][k];rows.append(label+f" & {a['mean_delta']*factor:+.4f} & [{a['bootstrap95'][0]*factor:+.4f}, {a['bootstrap95'][1]*factor:+.4f}]")
 write('r6_paired','lrr',r'r6 minus r4 & Difference & 95\% log interval',rows)
 r7path=ROOT/'evidence/open_joint_r7_analysis.json'
 if r7path.exists():
  d7=json.loads(r7path.read_text());rows=[]
  for k,label,factor in [('hit_rate','Hit (pp)',100),('early_rate','Early (pp)',100),('miss_rate','Missing (pp)',100),('free_intrusion_m','Free (m)',1),('surface_distance_m','Distance (m)',1),('surface_recall_02','Recall (pp)',100)]:
   a=d7['paired_final_minus']['first_surface_r6']['development'][k];rows.append(label+f" & {a['mean_delta']*factor:+.4f} & [{a['bootstrap95'][0]*factor:+.4f}, {a['bootstrap95'][1]*factor:+.4f}]")
  write('r7_paired','lrr',r'r7 minus r6 & Difference & 95\% log interval',rows)
 j=json.loads((ROOT/'evidence/population_joint_r5_analysis.json').read_text());d=json.loads((ROOT/'evidence/first_surface_r6_analysis.json').read_text());values=[]
 for label,key in [('LiDAR PCA','lidar_pca'),('VGGT native fusion','native_fusion'),('AdaPoinTr (task-tuned)','adapointr_r2'),('CAPA-style TTA','capa_r2')]:values.append((label,j['stages'][key]['development']))
 for label,key in [('LiDAR R8','lidar_r8'),('Open charts r3','open_charts_r3'),('Ray attraction r4','ray_support_r4'),('First surface r6','final')]:values.append((label,d['stages'][key]['development']))
 r7=ROOT/'evidence/open_joint_r7_analysis.json'
 if r7.exists():values.append(('Joint r7',json.loads(r7.read_text())['stages']['final']['development']))
 rows=[]
 for label,s in values:rows.append(label+' & '+' & '.join(f"{s[k]['mean']*fac:.{dig}f}" for k,fac,dig in [('hit_rate',100,2),('early_rate',100,2),('miss_rate',100,2),('free_intrusion_m',1,3),('surface_distance_m',1,3),('surface_recall_02',100,2)]))
 write('main','lrrrrrr',r'Method & Hit $\uparrow$ & Early $\downarrow$ & Miss $\downarrow$ & Free $\downarrow$ & Dist. $\downarrow$ & Recall $\uparrow$',rows)
architecture();oracle();pruning();slices();tables()
if R7:
 fig,ax=plt.subplots(figsize=(4.6,3.1));plane_slice('Joint-r7',ax);save(fig,'r7_failure_slice')
