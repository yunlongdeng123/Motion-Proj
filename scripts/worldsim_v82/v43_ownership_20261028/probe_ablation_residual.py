"""After-removal class contribution, using the same frozen row labels as the ablation.

No relabeling from edited RGB and no claimant that center depth is a surface hit.
"""
import json,time
from pathlib import Path
import numpy as np
from PIL import Image
from train_actor import Actor,NativeMotion,ROOT,OLD,torch,cpu,prepare_rasterization,helpers
from gpu_slot import exclusive
from v43_resource_control import require_start
from semantic_ablation_rules import classify_rows,removal_mask
from segmentation import write,NAMES,COLORS

def main():
    ctx='14705303724557273004_3105_000_3125_000';require_start(ctx);R=ROOT/'first_gpu_review_v1';out=R/'residual_class_probe';out.mkdir(exist_ok=False)
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);require_start(ctx);ds=ROOT/'datasets'/ctx;frames=json.loads((ds/'manifest.json').read_text())['frames'];actor=Actor(np.load(ds/'seeds.npz'));actor.load_state_dict(torch.load(str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor']);n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);ns=len(n.static._xyz);quarantine=json.loads((ROOT/'cpu_review_20261028/unknown_candidates.json').read_text())['ids'];records=[]
        for raw in [98,101,113,114]:
            i=next(i for i,f in enumerate(frames) if f['raw']==raw);f=frames[i];ob=np.load(ds/('%03d.npz'%raw));body=ob['body'];cam=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{});st=prepare_rasterization(cam,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);dy=actor.kwargs(n,cam,f);st['features']=st['features'][:,:2];dy['features']=dy['features'][:,:2];kw=helpers.merge_rows(st,dy);observed=np.load(ROOT/'boundary_semantics_v2'/ctx/('actor10k_%03d.npz'%raw));env=n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
            for radius in [0,5]:
                labels=classify_rows(observed,len(kw['means3D']),ns,quarantine,radius,.2);removed=removal_mask(labels,'hide_both');query={**kw,'opacities':kw['opacities'].clone()};query['opacities'][torch.from_numpy(removed).cuda()]=0
                classes=torch.from_numpy(labels.astype('int64')).cuda();onehot=torch.nn.functional.one_hot(classes,num_classes=20).float();query['features']=torch.cat((kw['features'],onehot),1);result=helpers.raster(cam,n.args,n.static.active_sh_degree,query,n.bg);mass=cpu(result['extra']);alpha=cpu(result['alpha'][0]);rgb=cpu(result['rgb']+(1-result['alpha'])*env);saved=np.asarray(Image.open(R/'semantic_ablation'/(str(radius)+'px/hide_both')/('%03d.png'%raw)));difference=int(np.abs(np.round(np.clip(rgb.transpose(1,2,0),0,1)*255).astype('int16')-saved.astype('int16')).max());assert difference<=1
                weights=mass[:,body].sum(1);label=mass.argmax(0).astype('uint8');label[alpha<.01]=0;Image.fromarray(COLORS[label]).save(out/('%03d_%dpx_classes.png'%(raw,radius)))
                valid=np.isfinite(ob['depth'])&body;records.append(dict(raw=raw,radius=radius,class_alpha_mass_in_original_target_mask={NAMES[k]:float(w) for k,w in enumerate(weights) if w>0},remaining_composite_alpha_in_target_mask=float(alpha[body].mean()),RGB_auxiliary_byte_max_difference=difference,original_target_LiDAR_median_m=float(np.median(ob['depth'][valid])) if valid.any() else None,remaining_alpha_weighted_center_depth_m=float(cpu(result['depth_sum'][0])[body].sum()/max(alpha[body].sum(),1e-8)/n.scale),no_pure_vehicle_identity_claim=True))
        assert n.unchanged();write(out/'summary.json',dict(records=records,training_updates=0,labels='carried original/no-edit observed semantics, not recomputed from edited RGB',center_depth_not_surface_hit=True,shadow_not_reassigned=True,quality_accepted=False,human_verdict=None));print(json.dumps(records),flush=True)

if __name__=='__main__':main()
