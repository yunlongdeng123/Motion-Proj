"""Common-observation evaluation, zero-update and time/amplitude controls."""
import argparse,json,sys,time,gc,math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from segmentation import ROOT,OLD,CONTEXTS,write
from train_actor import Actor,render,evaluate,NativeMotion,torch,cpu,prepare_rasterization,helpers
from gpu_slot import exclusive
def save(p,t):p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((np.clip(cpu(t).transpose(1,2,0),0,1)*255).round().astype('uint8')).save(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--context',action='append');p.add_argument('--output',required=True);p.add_argument('--checkpoint');a=p.parse_args();dest=ROOT/a.output;dest.mkdir(exist_ok=False)
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1)
        for ctx in a.context or CONTEXTS:
            out=dest/ctx;out.mkdir();dataset=ROOT/'datasets'/ctx;manifest=json.loads((dataset/'manifest.json').read_text());frames=manifest['frames'];data=np.load(dataset/'seeds.npz');actor=Actor(data);step=0
            if a.checkpoint:
                path=ROOT/'actor_campaign_v1'/ctx/'formal10k'/a.checkpoint;s=torch.load(str(path),map_location='cuda');actor.load_state_dict(s['actor']);step=s['step'];del s
            n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);obs=[]
            for f in frames:
                d=np.load(dataset/('%03d.npz'%f['raw']));obs.append((torch.from_numpy(d['rgb'].transpose(2,0,1).astype(np.float32)/255).cuda(),torch.from_numpy(d['body']).cuda(),torch.from_numpy(d['depth']).cuda()))
            evaluate(out,n,actor,frames,obs,step);rows=[]
            for i,f in enumerate(frames):
                camera=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{});gt,body,depth=obs[i]
                original=n.render(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time'],dynamic_query_model_time=f['model_time']),dict(family='no_edit'))
                new=render(n,actor,f,i);isonew=render(n,actor,f,i,static=False)
                kd=prepare_rasterization(camera,n.dynamic,n.args,dynamic=True);kd['means2D']=torch.zeros_like(kd['means3D']);isoold=helpers.raster(camera,n.args,n.dynamic.active_sh_degree,kd,n.bg)
                source_image=np.array(Image.open(OLD/'motion_scene_inputs_v1'/ctx/'image/image_0'/(f['frame_id']+'.png')).convert('RGB'))/255.
                native_gt=torch.from_numpy(cv2.resize(source_image,(480,320)).astype(np.float32).transpose(2,0,1)).cuda()
                def score(rgb,reference=gt):
                    e=(rgb-reference).square();return dict(full_psnr=-10*math.log10(max(float(e.mean()),1e-12)),body_psnr=-10*math.log10(max(float(e[:,body].mean()),1e-12)))
                valid=torch.isfinite(depth)&body;dmetrics={}
                for tag,iso in [('old',isoold),('new',isonew)]:
                    active=valid&(iso['alpha'][0]>.1);expected=iso['depth_sum'][0]/iso['alpha'][0].clamp_min(1e-6)/n.scale;errors=(expected[active]-depth[active]).abs()
                    dmetrics[tag]=dict(actual_return_pixels=int(valid.sum()),covered_return_pixels=int(active.sum()),mean_absolute_z_error_m=float(errors.mean()) if len(errors) else None,median_absolute_z_error_m=float(errors.median()) if len(errors) else None,within_half_meter_fraction=float((errors<.5).float().mean()) if len(errors) else None)
                rows.append(dict(raw=f['raw'],old=score(original['rgb']),new=score(new['rgb']),native_preprocess_old=score(original['rgb'],native_gt),native_preprocess_new=score(new['rgb'],native_gt),sparse_depth=dmetrics,new_body_conditioned_alpha_mae=float((isonew['alpha'][0][body]-1).abs().mean()),new_outside_body_alpha_mass=float(isonew['alpha'][0][~body].sum())))
                if i in [9,24,39]:save(out/'paired'/('old_%03d.png'%f['raw']),original['rgb']);save(out/'paired'/('new_%03d.png'%f['raw']),new['rgb'])
            controls=[]
            for i in [9,16,24,32,39]:
                for yaw in [0,7.5,15]:
                    E=Rotation.from_euler('z',yaw,degrees=True).as_matrix();r=render(n,actor,frames[i],i,edit=dict(rotation=E));save(out/'time_amplitude'/('raw%03d_yaw%04.1f.png'%(frames[i]['raw'],yaw)),r['rgb']);controls.append(dict(raw=frames[i]['raw'],yaw_deg=yaw,camera='same observed camera at this time',rigid_geometry=True))
            replay=[]
            for i in [39,24,9]:
                r=render(n,actor,frames[i],i,edit=dict(rotation=Rotation.from_euler('z',15,degrees=True).as_matrix()));path=out/'time_amplitude'/('raw%03d_yaw15.0.png'%frames[i]['raw']);ref=np.array(Image.open(path));current=(np.clip(cpu(r['rgb']).transpose(1,2,0),0,1)*255).round().astype('uint8');assert np.array_equal(ref,current);replay.append(dict(raw=frames[i]['raw'],reverse_order_byte_difference=0))
            assert n.unchanged();write(out/'common_observation_comparison.json',dict(context=ctx,step=step,frames=rows,time_amplitude_controls=controls,reverse_replay=replay,same_RGB_resize_and_depth_protocol=True,training_preprocessing='PIL LANCZOS then RGB8 /255, differs from native',native_evaluation_preprocessing='np.array(PIL RGB)/255 float64 -> cv2.INTER_LINEAR -> float32, exact source recipe',training_updates=0,original_checkpoint_unchanged=True,quality_accepted=False,human_verdict=None))
            del n,actor;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
