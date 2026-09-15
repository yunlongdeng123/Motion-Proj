import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;D=W/'evidence_milestone3';O=W.parents[1]/'outputs/Simulation_Impact_Research';C=D/'coverage_controls'
a=json.loads((C/'pdm_simulation_summary.json').read_text());full=[r for r in json.loads((D/'pdm_simulation_summary.json').read_text()) if r['scene']=='scene-0061' and r.get('protocol')=='build_scale' and r['condition']=='full']
groups=[('Real LiDAR baseline','real','baseline'),('GT restricted to camera FOV','GT_FOV_only','coverage'),('GT without common missing beams','GT_without_common_missing','coverage'),('Full reconstructed scans',None,'full'),('GT: remove inside-FOV misses only','missing_inside_fov_only','coverage'),('GT: remove outside-FOV misses only','missing_outside_fov_only','coverage'),('Full + restore outside-FOV misses','full_restore_outside_fov','oracle'),('Full + restore inside-FOV misses','full_restore_inside_fov_missing','oracle'),('Full + restore all missing beams','full_restore_all_missing','oracle'),('GT + cone-local model errors only','cone_only_error','local'),('Full + restore cone returns','full_restore_cones','local')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(12,7));colors={'baseline':'#267caa','coverage':'#7895a8','full':'#cf4946','oracle':'#b39964','local':'#4d967f'}
for i,(label,key,kind) in enumerate(groups):
 r=full if key is None else [r for r in a if r['condition']==key];n=len(r);k=sum(x['actor_overlap_any'] for x in r);ax.barh(i,k/n,color=colors[kind],height=.65);ax.text(k/n+.02,i,f'{k} / {n}',va='center',fontsize=11)
ax.set_yticks(range(len(groups)),[g[0] for g in groups]);ax.invert_yaxis();ax.set(xlim=(0,1.16),xticks=[0,.25,.5,.75,1],xticklabels=['0%','25%','50%','75%','100%'],xlabel='Executions with a recorded cone-box overlap');ax.tick_params(axis='y',length=0)
fig.suptitle('F10  A shared coverage limit explains most of the candidate',x=.025,ha='left',fontsize=18,weight='bold');fig.text(.025,.87,'Scene 0061 · 59 additional official policy + PDM executions · unchanged RGB, causal ego state and vehicle model',fontsize=11,color='#53657b')
fig.text(.025,.025,'Restore/hybrid controls use oracle real LiDAR. Camera FOV does not guarantee occlusion visibility.\nThe real-LiDAR baseline has only 1.6 cm clearance; contact counts are not proof of severe crashes or intrinsic phantom generation.',fontsize=11,color='#53657b')
fig.subplots_adjust(left=.37,right=.94,top=.81,bottom=.17)
for ext in ['png','pdf','svg']:fig.savefig(O/f'F10_Coverage_And_Local_Restoration.{ext}',dpi=210,bbox_inches='tight',facecolor='white')
print('Saved F10')
