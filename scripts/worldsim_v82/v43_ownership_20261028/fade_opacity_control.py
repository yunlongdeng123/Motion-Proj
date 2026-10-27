"""Only optical density changes in this read-only fade ablation; no model writeback."""
import json,time,os
from pathlib import Path
import numpy as np
from train_actor import Actor,NativeMotion,render,prepare_rasterization,helpers,torch,cpu,ROOT,OLD
from segmentation import CONTEXTS,write
from diagnose_first_badcases import save
from v43_resource_control import require_start
from gpu_slot import exclusive
from v43_entry import render_query

def main():
    ctx=CONTEXTS[0];require_start(ctx);out=ROOT/'first_fade_control_v1';out.mkdir(exist_ok=False);status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',training_updates=0)
    def update(**kw):status.update(kw);write(out/'status.json',status)
    update()
    with exclusive(),torch.no_grad():
        torch.set_num_threads(1);started=time.monotonic();ds=ROOT/'datasets'/ctx;frames=json.loads((ds/'manifest.json').read_text())['frames'];actor=Actor(np.load(ds/'seeds.npz'));actor.load_state_dict(torch.load(str(ROOT/'actor_campaign_v1'/ctx/'formal10k/checkpoint010000.pth'),map_location='cuda')['actor']);n=NativeMotion(OLD/'native_scene_training_v1'/ctx/'formal30k',30000);before={k:v.clone() for k,v in actor.state_dict().items()};rows=json.loads((OLD/'motion_edit_suite_v1'/ctx/'turn_left15.json').read_text())['frames'];records=[];bypass_max=0
        from render_motion_suite import actual_native_render
        for row in rows:
            i=row['camera_index'];f=frames[i];raw=f['raw'];camera=n.camera(row,{});baseline=render_query(n,row,dict(family='turn'),actor,f);isolated=render(n,actor,f,i,static=False,edit=dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']))
            st=prepare_rasterization(camera,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);st['features']=torch.cat((st['features'],torch.zeros_like(st['features'][:,:1])),1)
            dy=actor.kwargs(n,camera,f,dict(rotation=row['target_rotation_delta_world'],center=row['target_requested_center_world']));dy['opacities']=1-(1-dy['opacities']).square();kw=helpers.merge_rows(st,dy);changed=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg);env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1);changed['rgb']=changed['rgb']+(1-changed['alpha'])*env
            iso2=helpers.raster(camera,n.args,n.static.active_sh_degree,dy,n.bg);save(out/'density2_turn_left15/native_rgb'/('%03d.png'%raw),changed['rgb'])
            np.savez_compressed(out/('%03d_maps.npz'%raw),alpha=cpu(changed['alpha'][0]).astype('float16'),dynamic_weight=cpu(changed['extra'][0]).astype('float16'),isolated_alpha=cpu(iso2['alpha'][0]).astype('float16'),visibility_ratio=cpu(changed['extra'][0]/iso2['alpha'][0].clamp_min(1e-8)).clip(0,1).astype('float16'),depth_m=cpu(changed['depth_sum'][0]/changed['alpha'][0].clamp_min(1e-8)/n.scale).astype('float16'))
            domain=(isolated['alpha'][0]>.1)|(iso2['alpha'][0]>.1);records.append(dict(raw=raw,base_alpha_mean=float(isolated['alpha'][0][domain].mean()),density2_alpha_mean=float(iso2['alpha'][0][domain].mean()),base_visible_mass=float(baseline['extra'][0].sum()),density2_visible_mass=float(changed['extra'][0].sum())))
            if raw in [98,101,109,113,114]:save(out/'density2_turn_left15/actor_rgb'/('%03d.png'%raw),iso2['rgb'])
            original=actual_native_render(camera,n.static,n.dynamic,n.args,n.bg,env_map=n.env)['render'];routed=render_query(n,row,dict(family='no_edit'));bypass_max=max(bypass_max,float((original-routed['rgb']).abs().max()))
        assert n.unchanged() and all(torch.equal(before[k],v) for k,v in actor.state_dict().items()) and bypass_max<2e-6
        write(out/'summary.json',dict(records=records,opacity_rule='alpha_new=1-(1-alpha_old)^2, diagnostic only',same_point_count=True,same_DC_color=True,same_scale=True,same_pose=True,checkpoint_unchanged=True,main_method_changed=False,no_edit_V1_routing_max_difference=bypass_max,training_updates=0,quality_accepted=False,human_verdict=None));update(phase='complete',wall_seconds=time.monotonic()-started)
if __name__=='__main__':main()
