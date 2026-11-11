"""固定输入域上比较旧/新KEEP与DELETE；代理参考不称隐藏GT。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'waymo_val065_delete30k_20261109_r1';O=C/'waymo_val065_handoff_fix_20261111_r1';E=O/'evaluation';E.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text());m=load(O/'final_manifest.json');prov={}
for arm,root in [('before',R),('after',O)]:
    for state in ['final_keep','delete_white']:prov[arm,state]={r['frame_key']:r for r in load(root/state/'provenance.json')['frames']}
    if arm=='after':
        for state in ['initial_keep','initial_delete']:prov[arm,state]={r['frame_key']:r for r in load(root/state/'provenance.json')['frames']}
bank=Path(load(O/'world_road/receipt.json')['source_bank']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
def arrays(arm,state,key):
    root=R if arm=='before' else O
    return np.load(root/state/prov[arm,state][key]['files']['float_npz'])
def linear(x):
    x=x.clip(0,1);return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def cell(error,mask):return dict(count=int(mask.sum()),sum_abs=float(error[:,mask].sum(dtype=np.float64)),mean=float(error[:,mask].mean(dtype=np.float64)) if mask.any() else None)
rows=[];alpha_max=0
for frame in m['frames']:
    raw=frame['key'];lab=np.asarray(Image.open(frame['instance_mask']).resize((480,320),Image.Resampling.NEAREST));target=lab==10;other=(lab>0)&~target
    with np.load(R/'world_road/views'/f'{raw:03d}.npz') as v:
        regions=dict(target=target,other_vehicles=other,marking=v['marking'],outside_target=~target,visible_road=v['observed_road'],real_target=target&v['real_domain'],inferred_target=target&v['inferred_domain']);reference=v['reference_linear'].copy()
    d=load(bank/'observations'/f'{raw:03d}.json');observed=np.asarray(Image.open(bank/d['image_path']).resize((480,320),Image.Resampling.BILINEAR),np.float32).transpose(2,0,1)/255
    oldkeep=arrays('before','final_keep',raw);newkeep=arrays('after','final_keep',raw);old=arrays('before','delete_white',raw);new=arrays('after','delete_white',raw)
    row=dict(raw=raw,regions={})
    for name,mask in regions.items():
        row['regions'][name]=dict(keep_before_observed=cell(abs(oldkeep['rgb']-observed),mask),keep_after_observed=cell(abs(newkeep['rgb']-observed),mask),
            keep_change=cell(abs(newkeep['rgb']-oldkeep['rgb']),mask),delete_change=cell(abs(new['rgb']-old['rgb']),mask),
            delete_before_proxy=cell(abs(linear(old['rgb'])-reference),mask) if name in ['real_target','inferred_target'] else None,
            delete_after_proxy=cell(abs(linear(new['rgb'])-reference),mask) if name in ['real_target','inferred_target'] else None,
            delete_before_group_alpha_T=old['group_alpha_T'][:,mask].mean(1).tolist() if mask.any() else None,
            delete_after_group_alpha_T=new['group_alpha_T'][:,mask].mean(1).tolist() if mask.any() else None)
    value=float(abs(new['owner_alpha_T'][list(new['owner_alpha_actor_ids']).index(10)]).max());alpha_max=max(alpha_max,value);row['target_owned_alpha_max']=value;rows.append(row)
    if raw in [60,119,150,198]:
        images=[observed,oldkeep['rgb'],newkeep['rgb'],old['rgb'],arrays('after','initial_delete',raw)['rgb'],new['rgb']]
        labels=['Observed FRONT','Before: editable KEEP','After: editable KEEP','Before: DELETE','Expanded candidates, step0','After: DELETE, step1000']
        board=Image.new('RGB',(1440,724),'#0b111b');draw=ImageDraw.Draw(board)
        for i,(a,label) in enumerate(zip(images,labels)):
            x=i%3*480;y=i//3*362;draw.text((x+8,y+7),label,font=font,fill='white');board.paste(Image.fromarray(np.uint8(a.clip(0,1).transpose(1,2,0)*255+.5)),(x,y+34))
        board.save(E/f'comparison{raw}.png')
        for name,a in [('before_delete',old['rgb']),('after_delete',new['rgb']),('after_keep',newkeep['rgb'])]:Image.fromarray(np.uint8(a.clip(0,1).transpose(1,2,0)*255+.5)).save(E/f'{name}{raw}.png')
    if raw%30==0:print('EVALUATE',raw,flush=True)
aggregate={}
for name in rows[0]['regions']:
    aggregate[name]={}
    for metric in ['keep_before_observed','keep_after_observed','keep_change','delete_change','delete_before_proxy','delete_after_proxy']:
        values=[r['regions'][name][metric] for r in rows if r['regions'][name][metric] is not None]
        if values:
            n=sum(x['count'] for x in values);v=sum(x['sum_abs'] for x in values);aggregate[name][metric]=dict(pixel_frames=n,mean=v/(3*n) if n else None)
assert alpha_max<1e-6
result=dict(task='WAYMO-V44-HANDOFF-FIX-20261111',rows=rows,aggregate=aggregate,target_owned_alpha_max=alpha_max,
    reference_role='existing BUILD road plane/color proxy; not independent hidden-road truth',all_frames_training=True,human_verdict=None)
(E/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(E/'summary.json').write_text(json.dumps(dict(aggregate=aggregate,target_owned_alpha_max=alpha_max),indent=2)+'\n');print(json.dumps(aggregate,indent=2))
