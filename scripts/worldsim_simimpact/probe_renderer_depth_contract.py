"""用已知z=10m的单高斯检验官方模式的实际深度量纲。"""
import json
from pathlib import Path
import torch
from gsplat.rendering import rasterization
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
torch.set_num_threads(4)
def t(x):return torch.tensor(x,device='cuda',dtype=torch.float32)
rows=[]
for opacity in [.2,.5,.9]:
    common=dict(means=t([[0,0,10]]),quats=t([[1,0,0,0]]),scales=t([[1,1,.1]]),opacities=t([opacity]),colors=t([[.5,.5,.5]]),
        viewmats=torch.eye(4,device='cuda')[None],Ks=t([[[64,0,32],[0,64,32],[0,0,1]]]),width=64,height=64,packed=False,backgrounds=t([[0,0,0]]))
    baseline,ba,_=rasterization(**common,render_mode='RGB+ED')
    native,na,_=rasterization(**common,render_mode='RGB+ED+S',smts=torch.zeros((1,1,20),device='cuda'))
    y=x=32;alpha=float(na[0,y,x,0]);raw=float(native[0,y,x,3]);ed=float(baseline[0,y,x,3])
    rows.append({'known_z_m':10.,'opacity':opacity,'pixel_alpha':alpha,'RGB_ED_depth_m':ed,'RGB_ED_S_raw_depth_channel':raw,'RGB_ED_S_divided_by_alpha_m':raw/alpha,
        'observation':'RGB+ED+S emits accumulated depth in the installed official fork; label does not ensure expectation normalization'})
(R/'renderer_depth_contract.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
