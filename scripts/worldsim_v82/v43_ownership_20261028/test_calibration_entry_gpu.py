"""Real rendering interface closure, with original no-edit bypass and generic field."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from train_actor import Actor,NativeMotion,ROOT,OLD,torch,cpu,render
from gpu_slot import exclusive
from opacity_calibration import attach_saved_field,alpha_supervision
from v43_entry import render_query
from v43_resource_control import require_start
from segmentation import write

def main():
    ctx='14705303724557273004_3105_000_3125_000';require_start(ctx);R=ROOT/'first_gpu_review_v1'
    with exclusive(),torch.no_grad():
        ds=ROOT/'datasets'/ctx;frames=json.loads((ds/'manifest.json').read_text())['frames'];actor=Actor(np.load(ds/'seeds.npz'));actor.load_state_dict(torch.load(str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor']);n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);state=torch.load(str(R/'opacity/field/formal/checkpoint010000.pth'),map_location='cuda')['field'];wrapped=attach_saved_field(actor,state);field=wrapped.calibration
        row=next(r for r in json.loads((OLD/'motion_edit_suite_v1'/ctx/'turn_left15.json').read_text())['frames'] if r['raw']==113);f=frames[row['camera_index']];routed=render_query(n,row,dict(family='turn'),actor,f,calibration=field);direct=render(n,wrapped,f,row['camera_index'],edit=dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']));difference=float((routed['rgb']-direct['rgb']).abs().max());assert difference<2e-6
        native=n.render(row,dict(family='no_edit'));bypass=render_query(n,row,dict(family='no_edit'),actor,f,calibration=field);bypass_diff=float((native['rgb']-bypass['rgb']).abs().max());assert bypass_diff<2e-6
        png=np.asarray(Image.open(R/'opacity/field/formal/evaluation/010000/turn_left15/rgb/113.png'));display=np.round(cpu(routed['rgb']).transpose(1,2,0).clip(0,1)*255).astype('int16');pngdiff=int(np.abs(display-png.astype('int16')).max());assert pngdiff<=1
        iso=render(n,wrapped,f,row['camera_index'],static=False);a=iso['alpha'][0];body=torch.from_numpy(np.load(ds/('%03d.npz'%f['raw']))['body']).cuda();old=.1*(a-body.float()).abs().mean();balanced=.05*(a[body]-1).abs().mean()+.05*a[~body].abs().mean();legacydiff=float((alpha_supervision(a,body,'legacy')-old).abs());balanceddiff=float((alpha_supervision(a,body,'balanced')-balanced).abs());assert legacydiff<1e-7 and balanceddiff<1e-7
        q=field.factors()[:,0];inactive=field.confidence[:,0]==0;assert bool((q[inactive]==1).all()) and n.unchanged()
        write(R/'calibration_entry_verification.json',dict(edited_interface_float_max=difference,main_no_edit_bypass_float_max=bypass_diff,formal_saved_RGB_byte_max=pngdiff,legacy_loss_max_difference=legacydiff,balanced_loss_max_difference=balanceddiff,unsupported_q_exactly_one=True,unsupported_rows=int(inactive.sum()),q_min=float(q.min()),q_max=float(q.max()),source_checkpoint_unchanged=True,training_updates=0))
if __name__=='__main__':main()
