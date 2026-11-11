from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');O=C/'waymo_val065_delete30k_gpu_diagnosis_20261111_r1';P=O/'review';F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
records=[]
for raw in [119,198]:
    z=np.load(O/'candidate_limit_v1'/f'{raw}_selected_zero.npz');final=np.load(O/'joint_v1'/f'{raw}_final.npz');c=np.load(C/'waymo_val065_delete30k_diagnosis_20261109_r1'/f'coverage{raw}.npz');m=c['target']
    arrays=[final['rgb'],z['rgb'],np.abs(final['rgb']-z['rgb'])];labels=['Published DELETE','All gate candidates exit (diagnostic)','Absolute difference, no boost']
    b=Image.new('RGB',(1440,354),'#0b111b');d=ImageDraw.Draw(b)
    for i,(a,label) in enumerate(zip(arrays,labels)):
        d.text((i*480+8,7),label,font=F,fill='white');im=Image.fromarray(np.uint8(a.clip(0,1).transpose(1,2,0)*255+.5));b.paste(im,(i*480,34))
    b.save(P/f'{raw}_candidate_limit.png')
    records.append(dict(raw=raw,target_pixels=int(m.sum()),RGB_change=float(np.abs(final['rgb'][:,m]-z['rgb'][:,m]).mean()),
        published_group_alpha_T=final['group_alpha_T'][:,m].mean(1).tolist(),intervened_group_alpha_T=z['group_alpha_T'][:,m].mean(1).tolist(),
        role='extreme private-row diagnostic: original selected static opacity=0, not a valid preservation or recovered scene'))
(P/'candidate_limit_metrics.json').write_text(json.dumps(dict(records=records,training_updates=0),indent=2)+'\n')
summary=json.loads((O/'delivery/diagnosis_summary.json').read_text());receipt=json.loads((O/'candidate_limit_v1/receipt.json').read_text());summary['raster_forward_calls']+=len(receipt['rows']);summary['successful_module_seconds']['candidate_limit_v1']=receipt['seconds'];summary['same_sort_RGB_sum_maxabs']=max(summary['same_sort_RGB_sum_maxabs'],max(r['RGB_additivity_maxabs'] for r in receipt['rows']))
(P/'diagnosis_summary_final.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(records,indent=2))
