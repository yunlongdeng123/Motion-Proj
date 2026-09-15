import json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
W=Path(__file__).resolve().parent;D=W/'evidence_milestone3';O=W.parents[1]/'outputs/Simulation_Impact_Research'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
select=json.loads((D/'selection_summary.json').read_text());status={r['scene']:r for r in json.loads((D/'causal_status_amendment.json').read_text())['rows']}
fig,axs=plt.subplots(2,3,figsize=(15,7.3))
for ax,(name,info) in zip(axs.ravel(),select.items()):
 p=D/name;im=np.array(Image.open(p/'front.jpg'));inp=json.loads((p/'input.json').read_text());ref=json.loads((p/'log_reference.json').read_text());v=inp['views'][0];cam=np.linalg.inv(np.array(v['world_from_ego_camera']))@np.array(v['world_from_camera']);K=np.array(v['intrinsics_original']);scale=im.shape[1]/v['original_wh'][0]
 ax.imshow(im)
 ids={l['instance'] for l in info['metadata_selection']['leads']}
 for a in ref['records'][0]['boxes']:
  if a['instance'] not in ids:continue
  b=np.array(a['box']);y=b[6];rot=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
  corners=np.array(list(itertools.product([-1,1],repeat=3)))*np.array([b[4],b[3],b[5]])/2;corners=corners@rot.T+b[:3];cp=(corners-cam[:3,3])@cam[:3,:3];uv=cp@K.T;uv=uv[:,:2]/uv[:,2:]*scale;lo=uv.min(0);hi=uv.max(0)
  ax.add_patch(Rectangle(lo,*(hi-lo),fill=False,lw=1.8,ec='#f3cb4e'))
 s=status[name];lead=info['metadata_selection']['leads'][0]
 ax.set_title(f'{name}  |  distinct log {list(select).index(name)+1}',loc='left',fontsize=12,weight='bold')
 ax.text(.015,.04,f'Lead center {lead["center_forward_m"]:.1f} m  ·  ego {s["velocity"][0]:.1f} m/s\nPast-derived acceleration {s["acceleration"][0]:+.2f} m/s²',transform=ax.transAxes,color='white',fontsize=10,bbox={'fc':'black','alpha':.6,'ec':'none','pad':4})
 ax.axis('off')
fig.suptitle('F07  Six natural interaction logs, selected before inspecting geometry or driving outputs',x=.035,ha='left',fontsize=17,weight='bold')
fig.text(.035,.025,'Yellow boxes: recorded lead-vehicle annotations. First qualifying window in scene/time order; no sorting by model error.\nDiscovery cohort with ordinary moving-ego / nearby-vehicle criteria; not an independent final test or guaranteed hazardous encounter.',fontsize=11,color='#53657b')
fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.12,wspace=.08,hspace=.2)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F07_Natural_Interaction_Cohort.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
print('Saved F07')
