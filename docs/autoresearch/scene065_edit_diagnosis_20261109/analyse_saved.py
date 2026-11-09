from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R=CAM/'waymo_val065_delete30k_20261109_r1';O=CAM/'waymo_val065_delete30k_diagnosis_20261109_r1'
load=lambda p:json.loads(p.read_text())
m=load(R/'final_manifest.json'); prov={n:{x['frame_key']:x for x in load(R/n/'provenance.json')['frames']} for n in ['semantic_delete','final_keep','delete_white']}
def arrays(n,k):return np.load(R/n/prov[n][k]['files']['float_npz'])
def png(n,k):return Image.open(R/n/prov[n][k]['files']['rgb_png']).convert('RGB')
def mean(a,mask):return float(a[...,mask].mean(dtype=np.float64)) if mask.any() else None
def heat(a):
    a=np.clip(a,0,1);return Image.fromarray(np.uint8(np.stack([a,.85*a,.2*a],-1)*255+.5))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
records=[]
for raw in [60,119,150,198]:
    c=np.load(O/f'coverage{raw}.npz');mask=c['target']; regions=dict(target=mask,fit=mask&c['fit'],outside=mask&~c['inside'],inside_unknown=mask&c['inside']&~c['fit'],real=mask&c['real'],inferred=mask&c['inferred'])
    row=dict(raw=raw,regions={})
    before=arrays('semantic_delete',raw);after=arrays('delete_white',raw);keep=arrays('final_keep',raw)
    for key,reg in regions.items():
        d=dict(pixels=int(reg.sum()),group_alpha={str(name):mean(after['group_alpha_T'][i],reg) for i,name in enumerate(after['group_names'])},
               branch_alpha={str(name):mean(after['branch_alpha_T'][i],reg) for i,name in enumerate(after['branch_alpha_names'])},
               before_branch_alpha={str(name):mean(before['branch_alpha_T'][i],reg) for i,name in enumerate(before['branch_alpha_names'])},
               keep_target_alpha=mean(keep['group_alpha_T'][0],reg),
               before_to_after_RGB_change=mean(np.abs(before['rgb']-after['rgb']),reg),environment_RGB_T=mean(after['env_map'],reg))
        row['regions'][key]=d
    row['alpha_sum_error']=float(np.max(abs(after['group_alpha_T'].sum(0)-after['alpha'][0])))
    records.append(row)
    images=[png('final_keep',raw),png('delete_white',raw),Image.open(O/f'coverage{raw}.png'),heat(after['branch_alpha_T'][0]),heat(after['branch_alpha_T'][1]),heat(after['group_alpha_T'][3])]
    labels=['Editable scene KEEP','DELETE white sedan','Recovery coverage','Static branch alpha*T','Dynamic branch alpha*T','Road alpha*T']
    board=Image.new('RGB',(1440,724),'#0b111b');draw=ImageDraw.Draw(board)
    for i,(im,label) in enumerate(zip(images,labels)):
        x=i%3*480;y=i//3*362;draw.text((x+10,y+7),label,font=font,fill='#e7edf6');board.paste(im,(x,y+34))
    board.save(O/f'evidence{raw}.png')
    for name in ['semantic_delete','final_keep','delete_white']:png(name,raw).save(O/f'{name}{raw}.png')
gate=torch.load(R/'gate1000/checkpoint001000.pt',map_location='cpu',weights_only=False)['gate'].sigmoid().numpy()
summary=dict(task='WAYMO-065-V43-V44-DIAGNOSIS-20261109',records=records,
 gate=dict(rows=len(gate),quantiles=np.quantile(gate,[0,.1,.5,.9,1]).tolist(),under_half=int((gate<.5).sum())),
 training_updates=0,shader_RGB_grouping_available=False,diagnosis_is_saved_joint_alpha_T_not_independent_layer_renders=True,
 GPU_unavailable=True,automatic_shutdown=False)
(O/'saved_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(gate=summary['gate'],keys=[dict(raw=r['raw'],regions={k:v for k,v in r['regions'].items() if k in ['outside','fit']}) for r in records]),indent=2))
