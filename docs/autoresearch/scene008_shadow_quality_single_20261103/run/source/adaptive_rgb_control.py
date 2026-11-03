"""Zero-update calibration-off control, fixed gate and refined representation."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import binary_dilation,binary_erosion
from PIL import Image
from local_runtime import Experiment,RUN
from editor_common import cpu,save
import r4_runtime as r4

def main():
    e=Experiment();ck=torch.load(RUN/'formal/checkpoint001000.pt',map_location='cuda',weights_only=False);e.repair.load_state_dict(ck['repair'])
    dest=RUN/'adaptive_rgb_control';dest.mkdir(exist_ok=False);rows=[];num=0.;denom=0.;cache=[]
    ref=e.ref;uncorrected=np.clip(ref['reference_linear']-ref['native_correction_linear'],0,1)
    uncorrected=np.where(uncorrected<=.0031308,12.92*uncorrected,1.055*np.maximum(uncorrected,1e-10)**(1/2.4)-.055)
    dc=torch.tensor((uncorrected-.5)/.28209479177387814,device='cuda',dtype=torch.float32)[:,None]
    with torch.no_grad():
        original=e.repair.road.dc.clone()
        for raw in range(89,138):
            on=cpu(e.render(raw)['rgb']).transpose(1,2,0);e.repair.road.dc.copy_(dc);off=cpu(e.render(raw)['rgb']).transpose(1,2,0);e.repair.road.dc.copy_(original)
            base=cpu(e.base(raw)).transpose(1,2,0);d=e.view(raw);target=d['reference_linear'];b=cpu(r4.linear(e.base(raw))).transpose(1,2,0)
            y=b@np.array([.2126,.7152,.0722]);ry=target@np.array([.2126,.7152,.0722]);dark=d['repair_mask']&(y<.78*np.maximum(ry,.01))&((ry-y)>.025)
            left=np.indices(y.shape)[1]<220;clean=d['repair_mask']&~binary_dilation(dark,iterations=5)&(d['actor_labels']==0)&(np.abs(b-target).max(2)<.035)
            # Match quality preservation exclusions.
            yellow=(base[:,:,0]>1.22*base[:,:,2])&(base[:,:,1]>1.08*base[:,:,2])&(base[:,:,0]>.32)&d['repair_mask']&(d['actor_labels']==0)
            clean&=~binary_dilation(yellow,iterations=2)
            delta=on-off;num+=float((delta[clean]*(base-off)[clean]).sum());denom+=float((delta[clean]**2).sum())
            clean&=left;cache.append((base[clean],off[clean],delta[clean]))
            row=dict(raw=raw,samples=int(clean.sum()),on_MAE=float(np.abs(on-base)[clean].mean()),off_MAE=float(np.abs(off-base)[clean].mean()));rows.append(row)
            if raw==105:save(dest/'105_on.png',torch.tensor(on.transpose(2,0,1),device='cuda'));save(dest/'105_off.png',torch.tensor(off.transpose(2,0,1),device='cuda'))
    den=sum(x['samples'] for x in rows)
    gain=float(np.clip(num/max(denom,1e-12),0,2.5));fitted=sum(float(np.abs(off+gain*delta-base).sum()) for base,off,delta in cache)/(3*den)
    result=dict(updates=0,same_gate_geometry_opacity=True,rows=rows,on_MAE=sum(x['on_MAE']*x['samples'] for x in rows)/den,off_MAE=sum(x['off_MAE']*x['samples'] for x in rows)/den,samples=den,parent_unchanged=e.unchanged(),least_squares_gain=gain,predicted_fitted_left_MAE=fitted,gain_fit_domain='all native-clean road, RGB residual least squares; all49 BUILD views')
    (dest/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
