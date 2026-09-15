import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from PIL import Image
B=pathlib.Path(__file__).resolve().parent;OUT=B.parents[1]/'outputs/V74_Main_Paper_Figures';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
A=json.loads((B/'evidence/aggregate.json').read_text());rows=A['per_actor'];table=A['table'];meta=json.loads((B/'official_case_assets/manifest.json').read_text())
METHODS=['vggt','omega512','dvgt1','pi3x'];LABELS={'vggt':'Official VGGT','omega512':'VGGT-Omega 512*','dvgt1':'DVGT-1','pi3x':'Pi3X'};colors={'vggt':'#3e64bd','omega512':'#9a57ac','dvgt1':'#b86b13','pi3x':'#168475'};red='#d93649';teal='#008577';ink='#172b46';gray='#7f8b9a'
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(F/(name+'.'+ext),dpi=220,facecolor='white')
 plt.close(fig)
def row(m,v,owner,p='cal_build_1.0'):return next((x for x in rows if x['method']==m and x['variant']==v and x['owner']==owner and x['protocol']==p),None)
def item(m,v,p='cal_build_1.0'):return next((x for x in table if x['method']==m and x['variant']==v and x['protocol']==p),None)
selected=[next(x for x in meta if x['actor']['owner']==owner) for owner in ['204704542f8642dc8ab046ffbd70e0c5','01215d6383f44ab48cb5e28618cb1064','cc1739871a724fe085eb6e3048d1008d','0d8428e135f54d55b724631c24f60e4e','ff0246ef7b354ee9b49c359455583681']]
fig,axes=plt.subplots(len(selected),5,figsize=(14,11));fig.subplots_adjust(left=.055,right=.99,top=.89,bottom=.11,hspace=.48,wspace=.08)
fig.text(.055,.967,'Where the reconstructed surface enters measured free space',fontsize=19,fontweight='bold',color=ink)
fig.text(.055,.939,'Same target and RGB crop across columns  |  12-image input  |  fixed calibrated-depth surface + BUILD background scale',fontsize=10.5,color=gray)
for i,rec in enumerate(selected):
 e=rec['actor'];p=B/'official_case_assets'/e['owner'];image=Image.open(p/'rgb.jpg');proj=np.load(p/'projection.npz');T=proj['camera_from_actor'];K=proj['K'];box=np.array(rec['bbox']);margin=max(35,(box[2]-box[0])*.12)
 def project(x):
  q=x@T[:3,:3].T+T[:3,3];z=q[:,2];q=q@K.T;return q[:,:2]/q[:,2:3],z
 for j,method in enumerate([None]+METHODS):
  ax=axes[i,j];ax.imshow(image)
  if j==0:
   ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,ec='#ffc400',lw=1.6))
   ax.text(.02,.03,f'{e["scene"]}\nGT rays: {e["heldout_actor_returns"]}',transform=ax.transAxes,color='white',fontsize=9,bbox=dict(fc=ink,alpha=.8,ec='none',pad=3))
  else:
   q=p/method/'twelve';r=row(method,'twelve',e['owner'])
   if r and (q/'primary_surface.npz').exists():
    mesh=np.load(q/'primary_surface.npz');ray=np.load(q/'cal_build_1.0_rays.npz');early=(ray['first']<ray['ranges']-.2)&ray['build_supported'];ids=np.unique(ray['face_ids'][early]);ids=ids[ids>=0]
    if len(ids):
     tri=mesh['vertices'][mesh['faces'][ids]];uv,z=project(tri.reshape(-1,3));uv=uv.reshape(-1,3,2);ok=(z.reshape(-1,3)>.1).all(-1);ax.add_collection(PolyCollection(uv[ok],facecolors=red,edgecolors=red,linewidths=.5,alpha=.80))
    c=r['counts'];sup=r['strata']['build_supported'];text=f'Hit {100*c["hit"]/c["rays"]:.1f}%  |  Early {100*c["early"]/c["rays"]:.1f}%\nSupported Early {sup["early"]}/{sup["rays"]}'
    ax.text(.5,-.05,text,ha='center',va='top',transform=ax.transAxes,fontsize=9,color=ink,linespacing=1.5)
   else:ax.text(.5,.5,'Pending weights / inference',transform=ax.transAxes,ha='center',color=ink,bbox=dict(fc='white',ec='none',alpha=.9))
  ax.set_xlim(max(0,box[0]-margin),min(image.width,box[2]+margin));ax.set_ylim(min(image.height,box[3]+margin*.65),max(0,box[1]-margin*.65));ax.set_aspect('auto');ax.axis('off')
  if i==0:ax.set_title('RGB + target' if method is None else LABELS[method],color=ink,fontweight='bold',pad=10)
fig.text(.055,.025,'Red = predicted triangles causing >0.20 m early hits on BUILD-supported heldout returns. Empty red regions are retained.\n4 distinct logs shown; the 5th log has too little reference support for a main visual example. *User-provided mirror of original 512 checkpoint.',fontsize=9.5,color=gray,linespacing=1.5)
save(fig,'fig02_cross_scene_official_models')
# 同一白车的观测与输出表面数量控制。
fig,axes=plt.subplots(1,3,figsize=(13,4.8));fig.subplots_adjust(left=.065,right=.98,top=.76,bottom=.23,wspace=.31)
variants=['six','twelve_common6','twelve'];metrics=['hit','early','near_vertices'];titles=['Correct first hits ↑','Early returns ↓','Surface-point recall ↑']
for ax,key,title in zip(axes,metrics,titles):
 for m in METHODS:
  rs=[row(m,v,'204704542f8642dc8ab046ffbd70e0c5') for v in variants]
  if any(x is None for x in rs):continue
  y=[100*x['counts'][key]/x['counts']['rays'] for x in rs];ax.plot(range(3),y,'o-',color=colors[m],lw=2,ms=5,label=LABELS[m])
 ax.set_xlim(-.12,2.5);ax.set_xticks([0,1,2],['6 in\n6 maps','12 in\ncommon 6 maps','12 in\n12 maps']);ax.set_title(title,loc='left',fontweight='bold',color=ink);ax.set_ylabel('% of the same 752 rays');ax.grid(axis='y',alpha=.17)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.9),ncol=4,frameon=False)
fig.text(.065,.935,'More input evidence and more output surfaces are different interventions',fontsize=17,fontweight='bold',color=ink)
fig.text(.065,.06,'White car, fixed before this run. Common-6 readout is a post-hoc ordinary control; all predictions are frozen.\nKnown cameras / actor poses + one global BUILD background scale. No QUERY fitting. Surface recall uses a 0.20 m vertex neighborhood.',fontsize=10,color=gray)
save(fig,'fig03_white_car_evidence_vs_surface_count')
# 每日志结果和真实分母；弱参考日志仍保留。
logs=sorted(set(x['log'] for x in rows));lognames={'5bd40b613ac740cd9dbacdfbc3d68201':'0919','8fefc430cbfa4c2191978c0df302eb98':'1089','ca6d14b008ed4e0bb6b1eaaedadbd6c1':'0519+0520','ddc03471df3e4c9bb9663629a4097743':'0359','f38ef5a1e9c941aabb2155768670b92a':'0048'}
fig,axes=plt.subplots(1,2,figsize=(12,5.3));fig.subplots_adjust(left=.14,right=.96,bottom=.2,top=.78,wspace=.34)
for ax,variant in zip(axes,['six','twelve']):
 arr=np.full((4,5),np.nan);den=np.zeros(5,int)
 for i,m in enumerate(METHODS):
  it=item(m,variant)
  if not it:continue
  for j,log in enumerate(logs):
   lr=next(x for x in it['logs'] if x['log']==log);arr[i,j]=100*lr['early'];den[j]=lr['rays']
 im=ax.imshow(arr,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
 for i in range(4):
  for j in range(5):ax.text(j,i,'—' if not np.isfinite(arr[i,j]) else f'{arr[i,j]:.1f}%',ha='center',va='center',fontsize=10,color='white' if arr[i,j]>55 else ink)
 ax.set_yticks(range(4),[LABELS[m] for m in METHODS]);ax.set_xticks(range(5),[lognames[l]+f'\nN={den[j]:,}' for j,l in enumerate(logs)],fontsize=9);ax.set_title('6 input / 6 output maps' if variant=='six' else '12 input / 12 output maps',loc='left',color=ink,fontweight='bold')
fig.text(.06,.94,'Cross-log first-return errors: show the denominator, not only the percentage',fontsize=16,fontweight='bold',color=ink);fig.text(.06,.875,'Early > 0.20 m after known calibration + BUILD background scale. 52 evaluable actors; 23 without owned QUERY returns remain undefined.',fontsize=10,color=gray);fig.text(.06,.055,'All 5 discovery logs retained. Log 0359 has only 25 actor rays and cannot establish robust failure prevalence.\nThese are fixed grid-surface diagnostics of official predictions, not native end-to-end LiDAR simulation benchmarks.',fontsize=10,color=gray)
save(fig,'fig04_cross_log_early_rates')
# 置信度控制；同口径尺度已固定，不据曲线挑参数。
fig,axes=plt.subplots(1,2,figsize=(11,4.7));fig.subplots_adjust(left=.08,right=.98,bottom=.22,top=.75,wspace=.30)
for ax,variant in zip(axes,['six','twelve']):
 for m in METHODS:
  pts=[]
  for keep in [1.,.9,.75,.5]:
   it=item(m,variant,'cal_build_'+str(keep))
   if it:pts.append((it['equal_log_mean']['near_vertices']*100,it['equal_log_mean']['early']*100,keep))
  if pts:
   ax.plot([p[0] for p in pts],[p[1] for p in pts],'o-',c=colors[m],lw=2,label=LABELS[m]);
   for x,y,k in [pts[0],pts[-1]]:ax.annotate(f'{int(k*100)}%',(x,y),xytext=(4,4),textcoords='offset points',fontsize=8,color=colors[m])
 ax.set_xlabel('Surface-point recall [%] ↑');ax.set_ylabel('Early returns [%] ↓');ax.set_title('6 input views' if variant=='six' else '12 input views',loc='left',fontweight='bold');ax.grid(alpha=.17)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.88),ncol=4,frameon=False);fig.text(.08,.925,'Does ordinary confidence filtering resolve the trade-off?',fontsize=17,fontweight='bold',color=ink);fig.text(.08,.055,'Per-view confidence retention fixed at 100 / 90 / 75 / 50%. Equal-log means, including weak-reference logs.\nCalibration and BUILD scale are identical within each curve. All points are reported; no threshold is selected after evaluation.',fontsize=10,color=gray)
save(fig,'fig05_confidence_controls')
print('comparison figures written',F)
