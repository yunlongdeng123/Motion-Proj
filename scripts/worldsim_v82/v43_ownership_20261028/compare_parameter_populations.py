"""Distinguish V42->V43 initialization differences from within-request mutation."""
import json
import numpy as np
from train_actor import Actor,NativeMotion,prepare_rasterization,ROOT,OLD,torch
from segmentation import CONTEXTS,write
from diagnose_first_badcases import stats
from gpu_slot import exclusive
from v43_resource_control import require_start

def main():
    ctx=CONTEXTS[0];require_start(ctx);dest=ROOT/'parameter_populations_v1.json';assert not dest.exists()
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);ds=ROOT/'datasets'/ctx;frames=json.loads((ds/'manifest.json').read_text())['frames'];actor=Actor(np.load(ds/'seeds.npz'));actor.load_state_dict(torch.load(str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor']);n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);rows=[]
        for raw in [98,101,109,113,114]:
            i=next(i for i,f in enumerate(frames) if f['raw']==raw);f=frames[i];cam=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{});old=prepare_rasterization(cam,n.dynamic,n.args,dynamic=True);new=actor.kwargs(n,cam,f)
            def population(k):
                mask=k['mask'].flatten().bool()&(k['opacities'].flatten()>.005);return dict(total_rows=len(k['means3D']),time_and_opacity_eligible_rows=int(mask.sum()),opacity=stats(k['opacities'][mask]),sigma_m=stats(k['scales'][mask]/n.scale))
            rows.append(dict(raw=raw,original=population(old),new=population(new)))
        assert n.unchanged();write(dest,dict(context=ctx,records=rows,scope='Within the original observed frame, all eligible rows; no semantic/vehicle-box filtering',new_representation='Measured6cm local seeds, DC-only appearance, rigidTrack, no densification',original_representation='Time-gated free per-point dynamic SH field',same_training_method=False,not_a_pure_constraint_comparison=True,training_updates=0))
if __name__=='__main__':main()
