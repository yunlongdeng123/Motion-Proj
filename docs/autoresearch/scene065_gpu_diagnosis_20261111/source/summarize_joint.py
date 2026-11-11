from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=C/'waymo_val065_delete30k_20261109_r1';O=C/'waymo_val065_delete30k_gpu_diagnosis_20261111_r1';P=O/'review';P.mkdir(exist_ok=True)
F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
def png(a):return Image.fromarray(np.uint8(np.clip(a.transpose(1,2,0),0,1)*255+.5))
def mean(a,mask):return float(a[...,mask].mean(dtype=np.float64)) if mask.any() else None
records=[];luma=np.array([.2126,.7152,.0722])
for raw in [60,119,150,198]:
    c=np.load(C/'waymo_val065_delete30k_diagnosis_20261109_r1'/f'coverage{raw}.npz');target=c['target'];regions=dict(target=target,fit=target&c['fit'],outside=target&~c['inside'],unknown=target&c['inside']&~c['fit'],real=target&c['real'],inferred=target&c['inferred'])
    data={mode:np.load(O/('interaction_v1' if mode=='gates_off_road_off' else 'joint_v1')/f'{raw}_{mode}.npz') for mode in ['final','gates_off','road_off','road_only','gates_off_road_off']};final=data['final'];reference=np.load(R/'world_road/views'/f'{raw:03d}.npz')['reference_linear']
    def linear(x):
        x=x.clip(0,1);return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    for mode,z in data.items():
        png(z['rgb']).save(P/f'{raw}_{mode}.png')
        rr={}
        for n,mask in regions.items():
            rr[n]=dict(pixels=int(mask.sum()),RGB_change_vs_final=mean(abs(z['rgb']-final['rgb']),mask),
                proxy_real_reference_linear_MAE=mean(abs(linear(z['rgb'])-reference),mask) if n=='real' else None,
                group_alpha_T=z['group_alpha_T'][:,mask].mean(1).tolist() if mask.any() else None,
                RGB_T={str(name):z['RGB_T'][i,:,mask].mean(0).tolist() for i,name in enumerate(z['RGB_names'])} if mask.any() else None)
        records.append(dict(raw=raw,mode=mode,regions=rr))
    # Direct RGB components have exactly the final sort; fixed exposure throughout.
    arrays=[final['rgb'],*list(final['RGB_T'][1:4]),final['environment_RGB_T']]
    labels=['Final DELETE','Static RGB*T','Dynamic RGB*T','Road RGB*T','Environment RGB*T']
    board=Image.new('RGB',(1440,724),'#0b111b');draw=ImageDraw.Draw(board)
    for i,(array,label) in enumerate(zip(arrays,labels)):
        x=i%3*480;y=i//3*362;draw.text((x+8,y+7),label,font=F,fill='white');board.paste(png(array),(x,y+34))
    board.save(P/f'{raw}_layers.png')
    modes=['gates_off_road_off','gates_off','road_off','final','road_only'];labels=['Semantic deletion baseline','Road added, gates off','Gates active, road off','Road + gates (published)','Road + environment only']
    board=Image.new('RGB',(1440,724),'#0b111b');draw=ImageDraw.Draw(board)
    for i,(mode,label) in enumerate(zip(modes,labels)):
        x=i%3*480;y=i//3*362;draw.text((x+8,y+7),label,font=F,fill='white');board.paste(png(data[mode]['rgb']),(x,y+34))
    board.save(P/f'{raw}_controls.png')
(P/'joint_metrics.json').write_text(json.dumps(dict(records=records,note='RGB_T are direct same-sort contributions; no automatic exposure; reference metric only in existing accepted source proxy',training_updates=0),indent=2)+'\n')
print(P)
