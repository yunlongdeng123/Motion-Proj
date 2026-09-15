"""局部传感器干预的恢复图；明确真实强度这一替代解释。"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/Simulation_Impact_Research';D=W/'evidence_native_detector'
rows=json.loads((D/'native_summary.json').read_text())+json.loads((D/'local_summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(14,6));fig.subplots_adjust(left=.08,right=.97,top=.69,bottom=.31,wspace=.25)
conditions=['','_local_real','_local_xyz','_local_intensity'];labels=['Original','Local real scan','Local position only','Local intensity only'];colors=['#ba6353','#3c8165','#5078a0','#9c7fba']
for ax,readout in zip(axs,['raw','median']):
    for j,t in enumerate(['0.3','0.5']):
        values=[sum(any(m['class']=='construction_vehicle' for m in r['scores'][t]['bev_iou0p5']['matches']) for r in rows if r['condition']==readout+c) for c in conditions]
        xx=np.arange(4)+(j-.5)*.34
        ax.bar(xx,values,width=.32,color=colors,alpha=.45 if j==0 else 1,hatch='//' if j==0 else None,edgecolor='white')
        for x,v in zip(xx,values):ax.text(x,v+.12,str(v),ha='center',fontsize=10,weight='bold')
    ax.set_xticks(range(4),labels,rotation=18,ha='right',fontsize=9);ax.set(ylim=(0,9),yticks=[0,2,4,6,8],ylabel='Matched excavator observations / 8');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_title(f'Native {readout}',weight='bold')
fig.suptitle('Local recovery separates a sensor failure from a geometry-method motivation',x=.03,ha='left',fontsize=17,weight='bold')
fig.text(.035,.865,'Same fitted asset → fixed 10-sweep scan → one local field intervention → same frozen CenterPoint',fontsize=12,color='#244a67')
fig.text(.035,.79,'Hatched bars: score ≥ 0.3    Solid bars: score ≥ 0.5    All matches require correct class and BEV IoU ≥ 0.5.',fontsize=10)
fig.text(.035,.18,'Local real replacement: 8/8 for both readouts. Median + intensity alone: 7/8 at score0.5; position-only: 2/8 raw, 3/8 median.',fontsize=11,weight='bold',color='#244a67')
fig.text(.035,.115,'GT box + 0.25m neighborhood; evaluator-held real scans are extra information. Position-only snaps predicted points to same-time real neighbors;',fontsize=10)
fig.text(.035,.075,'intensity-only keeps positions/counts fixed. These are sensor interventions, not edits to the reconstruction asset. No planning/safety recovery was measured.',fontsize=10)
fig.text(.035,.028,'Decision: retain the observed perception loss; do not promote this case as a pure geometry or phantom-surface Hero motivation.',fontsize=10,color='#244a67')
for ext in ['png','pdf','svg']:fig.savefig(O/f'F17_Local_Sensor_Recovery.{ext}',dpi=190,bbox_inches='tight')
