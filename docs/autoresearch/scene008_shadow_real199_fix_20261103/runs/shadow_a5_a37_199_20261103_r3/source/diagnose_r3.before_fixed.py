import json
from pathlib import Path
import torch
from r3_runtime import Experiment,RUN,save,road_kwargs,raster,write

@torch.no_grad()
def main():
    e=Experiment();ck=torch.load(RUN/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False);e.repair.load_state_dict(ck['repair'])
    dest=RUN/'diagnosis';dest.mkdir(exist_ok=True)
    o=e.render(105);save(dest/'105_final.png',o['rgb'])
    e.repair.gate_raw.fill_(-30);z=e.render(105);save(dest/'105_selected_zero.png',z['rgb'])
    c=e.camera(105);rr=road_kwargs(c,e.repair.road,e.n.args,precomputed_color=True)
    road=raster(c,e.n.args,0,rr,e.n.bg);save(dest/'105_road_only.png',road['rgb'])
    d=e.view(105);rows=[]
    for x,y in [(20,275),(40,280),(60,275),(90,255),(20,290),(80,285),(255,300),(260,290)]:
        rows.append(dict(x=x,y=y,mask=bool(d['repair_mask'][y,x]),
            rgb=o['rgb'][:,y,x].tolist(),gate_zero_rgb=z['rgb'][:,y,x].tolist(),
            road_alpha_T=float(o['extra'][0,y,x]),gate_zero_road_alpha_T=float(z['extra'][0,y,x]),
            standalone_alpha=float(road['alpha'][0,y,x]),target_linear=d['recovery_reference_linear'][y,x].tolist()))
    write(dest/'pixels.json',rows);print(json.dumps(rows))

if __name__=='__main__':main()
