import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from PIL import Image
B=pathlib.Path(__file__).resolve().parent;F=B.parents[1]/'outputs/V74_Main_Paper_Figures/figures';S=B/'secondary';A=json.loads((S/'summary.json').read_text());M=json.loads((B/'official_case_assets/manifest.json').read_text());ink='#172b46';red='#d93649';teal='#008577';gray='#697c8d'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(F/(name+'.'+ext),dpi=220,facecolor='white')
 plt.close(fig)
selected=[next(x for x in M if x['actor']['owner']==o) for o in ['204704542f8642dc8ab046ffbd70e0c5','01215d6383f44ab48cb5e28618cb1064','cc1739871a724fe085eb6e3048d1008d','0d8428e135f54d55b724631c24f60e4e']]
# 原生渲染和深度读出应有明确不同的输出名称。
fig,axes=plt.subplots(4,4,figsize=(13,9));fig.subplots_adjust(left=.055,right=.99,top=.875,bottom=.13,wspace=.06,hspace=.35)
for i,rec in enumerate(selected):
 e=rec['actor'];p=S/'cases'/e['owner']/'dggt_render';raw=Image.open(B/'official_case_assets'/e['owner']/'rgb.jpg');rend=Image.open(next(p.glob('*_rgb.png')));z=np.load(next(p.glob('*_depth_alpha.npz')));d=np.load(p/'native_depth.npz')['depth'];scale=np.percentile(d[np.isfinite(d)&(d>0)],95)
 for j,ax in enumerate(axes[i]):
  if j==0:ax.imshow(raw)
  elif j==1:ax.imshow(rend)
  else:ax.imshow(z['expected_depth'] if j==2 else d,vmin=0,vmax=scale,cmap='turbo')
  ax.set_aspect('auto');ax.axis('off')
  if i==0:ax.set_title(['Input RGB','DGGT native Gaussian RGB','Gaussian expected depth','Native depth head'][j],fontsize=11,fontweight='bold',color=ink)
 axes[i,0].text(.02,.04,e['scene'],transform=axes[i,0].transAxes,color='white',bbox=dict(fc=ink,ec='none',alpha=.8,pad=2))
fig.text(.055,.953,'DGGT: a rendered scene and a physical first return are different outputs',fontsize=16,fontweight='bold',color=ink)
fig.text(.055,.91,'Official nuScenes weights  |  12-image inputs  |  input-view reconstruction; no novel-view or closed-loop score',fontsize=10,color=gray)
fig.text(.055,.035,'Official mode-2 Gaussian rendering equations and sky background; predicted semantic sky mask replaces dataset GT mask.\nDepth colors share a scale within each row (native units). Expected depth is not an opaque first intersection.\nThe native depth and pose predictions exactly match VGGT in all 12 tested windows; these are not independent geometry failures.',fontsize=10,color=gray,linespacing=1.45)
save(fig,'fig07_dggt_native_rendering')
# 原先同一射线必须保留为反例。
owner='204704542f8642dc8ab046ffbd70e0c5';beam=382
methods=['vggt','omega512','dvgt1','pi3x','dvgt2','nksr']+(['noksr'] if 'noksr' in A else [])
labels={'vggt':'Official VGGT','omega512':'VGGT-Omega 512','dvgt1':'DVGT-1','pi3x':'Pi3X','dvgt2':'DVGT-2','nksr':'NKSR + BUILD LiDAR','noksr':'NoKSR + BUILD LiDAR'};vals=[-.322696464354];names=['Legacy adapted VGGT + LiDAR'];colors=[red]
for m in methods:
 p=(B/'official_case_assets'/owner/m/'twelve/cal_build_1.0_rays.npz') if m in methods[:4] else (S/'cases'/owner/m/'query_rays.npz' if m in ['nksr','noksr'] else S/'cases'/owner/m/'twelve/cal_build_1.0_rays.npz')
 if not p.exists():continue
 a=np.load(p);v=float(a['first'][beam]-a['ranges'][beam]);vals.append(v);names.append(labels[m]);colors.append(teal if abs(v)<=.2 else red)
fig,ax=plt.subplots(figsize=(11,5.4));fig.subplots_adjust(left=.30,right=.94,top=.78,bottom=.23)
ax.axvspan(-.2,.2,color=teal,alpha=.09);ax.axvline(0,color=gray,lw=1);ax.barh(range(len(vals)),vals,color=colors,height=.58);ax.set_yticks(range(len(vals)),names);ax.invert_yaxis();ax.set_xlabel('Predicted minus measured first-return range [m]');ax.set_xlim(min(-.45,min(vals)-.1),max(.28,max(vals)+.12));ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for i,v in enumerate(vals):
 if np.isfinite(v):ax.text(v+(.008 if v>=0 else -.008),i,f'{v:+.3f}',ha='right' if v<0 else 'left',va='center',fontsize=10,color=ink)
fig.text(.055,.935,'A retained counterexample: the original 0.323 m beam is fixed by all four primary models',fontsize=14,fontweight='bold',color=ink)
fig.text(.055,.865,'Same white-car beam #382, above the underbody band. Green band: the predeclared ±0.20 m Hit tolerance.',fontsize=10,color=gray)
fig.text(.055,.055,'The legacy adapted system is not official VGGT. Primary models use the declared camera / BUILD scale diagnostic.\nNKSR and NoKSR receive extra LiDAR points and normals. Different information budgets are shown, not ranked.\nThis successful beam is retained alongside the shared low-return failures.',fontsize=10,color=gray,linespacing=1.4)
save(fig,'fig08_legacy_beam_counterexample')
# 每个第二批方法的原生/适配读出，保留没有红色的面板。
cols=['dvgt2','nksr']+(['noksr'] if 'noksr' in A else [])
fig,axes=plt.subplots(4,len(cols)+1,figsize=(12,9),squeeze=False);fig.subplots_adjust(left=.055,right=.99,top=.86,bottom=.13,wspace=.08,hspace=.5)
for i,rec in enumerate(selected):
 e=rec['actor'];p=B/'official_case_assets'/e['owner'];image=Image.open(p/'rgb.jpg');proj=np.load(p/'projection.npz');T=proj['camera_from_actor'];K=proj['K'];box=np.array(rec['bbox']);margin=max(35,(box[2]-box[0])*.12)
 for j,m in enumerate([None]+cols):
  ax=axes[i,j];ax.imshow(image)
  if m is None:ax.add_patch(Rectangle(box[:2],box[2]-box[0],box[3]-box[1],fill=False,ec='#ffc400',lw=1.8));ax.text(.02,.03,e['scene'],transform=ax.transAxes,color='white',bbox=dict(fc=ink,ec='none',pad=2,alpha=.8))
  else:
   q=S/'cases'/e['owner']/m
   sf=q/'surface.npz' if m!='dvgt2' else q/'twelve/primary_surface.npz';rf=q/'query_rays.npz' if m!='dvgt2' else q/'twelve/cal_build_1.0_rays.npz'
   if sf.exists() and rf.exists():
    mesh=np.load(sf);r=np.load(rf);early=(r['first']<r['ranges']-.2)&r['build_supported'];ids=np.unique(r['face_ids'][early]);ids=ids[ids>=0]
    if len(ids):
     tri=mesh['vertices'][mesh['faces'][ids]];cp=tri.reshape(-1,3)@T[:3,:3].T+T[:3,3];uv=cp@K.T;uv=(uv[:,:2]/uv[:,2:3]).reshape(-1,3,2);ok=(cp[:,2].reshape(-1,3)>.1).all(-1);ax.add_collection(PolyCollection(uv[ok],fc=red,ec=red,lw=.3,alpha=.7))
    delta=r['first']-r['ranges'];text=f'Hit {np.sum(np.abs(delta)<=.2)}/{len(delta)}  |  Early {np.sum(delta<-.2)}/{len(delta)}';ax.text(.5,-.07,text,ha='center',va='top',transform=ax.transAxes,fontsize=10,color=ink)
   else:ax.text(.5,.5,'No usable method output',transform=ax.transAxes,ha='center',bbox=dict(fc='white',alpha=.8,ec='none'))
  ax.set_xlim(max(0,box[0]-margin),min(image.width,box[2]+margin));ax.set_ylim(min(image.height,box[3]+margin*.65),max(0,box[1]-margin*.65));ax.set_aspect('auto');ax.axis('off')
  if i==0:ax.set_title('RGB + target' if m is None else labels[m],fontsize=11,fontweight='bold',color=ink)
fig.text(.055,.946,'Second-batch reconstruction references: preserve input and representation differences',fontsize=15,fontweight='bold',color=ink)
fig.text(.055,.895,'DVGT-2: RGB → declared depth grid. NKSR / NoKSR: additional BUILD LiDAR + PCA normals → native learned surface.',fontsize=10,color=gray)
fig.text(.055,.035,'Same metadata-selected objects as the primary comparison. Red = native / adapted triangles causing >0.20 m early hits\non BUILD-supported heldout returns. Counts retain the full object QUERY denominator; red overlays show the supported stratum.\nPoint ownership uses a padded 3D annotation box, not semantic surface labels. These are discovery cases, not an independent test set.',fontsize=10,color=gray,linespacing=1.4)
save(fig,'fig09_secondary_cross_scene')
print('secondary figures generated',flush=True)
