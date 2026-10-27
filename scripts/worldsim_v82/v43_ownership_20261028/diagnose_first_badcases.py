"""Causal read-only probes for fade, ghost support, larger shifts and V1 bypass."""
import json,time,os,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from train_actor import Actor,NativeMotion,render,prepare_rasterization,helpers,torch,cpu,ROOT,OLD
from segmentation import CONTEXTS,write
from v43_resource_control import require_start
from gpu_slot import exclusive

CTX=CONTEXTS[0]
def save(p,t):
    p.parent.mkdir(parents=True,exist_ok=True);Image.fromarray((np.clip(cpu(t).transpose(1,2,0),0,1)*255).round().astype('uint8')).save(p)
def stats(t):
    a=cpu(t).ravel();return dict(min=float(a.min()),median=float(np.median(a)),p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),max=float(a.max()),mean=float(a.mean()))
def render_aux(n,actor,f,i,edit=None):
    camera=n.camera(dict(camera_index=i,frame_id=n.records[i]['img_name'],camera_model_time=f['model_time']),{})
    st=prepare_rasterization(camera,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);st['features']=torch.cat((st['features'][:,:2],st['features'].new_zeros((len(st['means3D']),4))),1)
    dy=actor.kwargs(n,camera,f,edit);color=(actor.dc[:,0]*.28209479177387814+.5).clamp_min(0);dy['features']=torch.cat((dy['features'][:,:2],color,torch.ones_like(color[:,:1])),1)
    kw=helpers.merge_rows(st,dy);out=helpers.raster(camera,n.args,n.static.active_sh_degree,kw,n.bg);env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1);out['rgb']=out['rgb']+(1-out['alpha'])*env
    return out
def static_without(n,row,mask):
    camera=n.camera(row,{});st=prepare_rasterization(camera,n.static,n.args);st['means2D']=torch.zeros_like(st['means3D']);st['opacities']=st['opacities'].clone();st['opacities'][mask]=0
    out=helpers.raster(camera,n.args,n.static.active_sh_degree,st,n.bg);env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1);out['rgb']=out['rgb']+(1-out['alpha'])*env;return out

def main():
    require_start(CTX);out=ROOT/'first_badcase_diagnostics_v1';out.mkdir(exist_ok=False);status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',training_updates=0,frames=0)
    def update(**kw):status.update(kw);write(out/'status.json',status)
    update()
    with exclusive(),torch.no_grad():
        require_start(CTX);torch.set_num_threads(1);started=time.monotonic();dataset=ROOT/'datasets'/CTX;frames=json.loads((dataset/'manifest.json').read_text())['frames'];actor=Actor(np.load(dataset/'seeds.npz'));p=torch.load(str(ROOT/'actor_campaign_v1'/CTX/'formal10k/checkpoint010000.pth'),map_location='cuda');actor.load_state_dict(p['actor']);del p
        n=NativeMotion(OLD/'native_scene_training_v1'/CTX/'formal30k',30000);before={k:v.clone() for k,v in actor.state_dict().items()};ns=len(n.static._xyz);geometry=actor.geometry();parameter_stats=dict(point_count=len(actor.xyz),opacity=stats(actor.opacity.sigmoid()),sigma_m=stats(actor.log_scale.exp()),dc=stats(actor.dc),original_dynamic_rows=len(n.dynamic._xyz),new_asset_reinitialized=True)
        specs={c:json.loads((OLD/'motion_edit_suite_v1'/CTX/(c+'.json')).read_text()) for c in ['no_edit','delete_target','translate_lateral_1m','turn_left15']};base_rows=specs['no_edit']['frames'];reference={str(r['raw']):r for r in base_rows};records=[];max_unchanged=0.;bypass_max=0.;delete_byte_max=0
        motion_status=json.loads((OLD/'motion_suite_execution_v1'/CTX/'status.json').read_text());old_suite=Path(motion_status.get('native_suite_path',str(OLD/'motion_suite_execution_v1'/CTX/'native_suite')))
        for case in ['no_edit','delete_target','translate_lateral_1m','turn_left15','translate_lateral_3m']:
            rows=specs['translate_lateral_1m' if case.endswith('3m') else case]['frames'];(out/case).mkdir()
            for row in rows:
                i=row['camera_index'];f=frames[i];raw=f['raw'];edit=None
                if case not in ['no_edit','delete_target']:
                    center=np.array(row['target_requested_center_world']);origin=np.array(f['actor_center_world'])
                    if case.endswith('3m'):center=origin+3*(center-origin)
                    edit=dict(rotation=row['target_rotation_delta_world'],center=center)
                if case=='delete_target':result=n.render(row,dict(family='delete'));iso=None;aux=None
                else:
                    result=render(n,actor,f,i,edit=edit);iso=render(n,actor,f,i,static=False,edit=edit);aux=render_aux(n,actor,f,i,edit);assert float((aux['rgb']-result['rgb']).abs().max())<2e-6
                save(out/case/'native_rgb'/('%03d.png'%raw),result['rgb'])
                if case=='no_edit':
                    bypass=n.render(row,dict(family='no_edit'));again=n.render(row,dict(family='no_edit'));bypass_max=max(bypass_max,float((again['rgb']-bypass['rgb']).abs().max()));save(out/'v1_no_edit'/'native_rgb'/('%03d.png'%raw),bypass['rgb'])
                if case=='delete_target':
                    source=old_suite/'delete_target/native_rgb'/('%03d.png'%raw)
                    if source.exists():delete_byte_max=max(delete_byte_max,int(np.abs(np.array(Image.open(source)).astype(np.int16)-np.array(Image.open(out/case/'native_rgb'/('%03d.png'%raw))).astype(np.int16)).max()))
                else:
                    A=iso['alpha'][0];W=aux['extra'][3];support=A>.1;inner=torch.from_numpy(binary_erosion(cpu(support),iterations=2)).cuda();den=A[inner].sum().clamp_min(1e-8);ratio=(W/A.clamp_min(1e-8)).clamp(0,1)
                    color=aux['extra'][:3];newrec=dict(case=case,raw=raw,points=len(actor.xyz),opacity=parameter_stats['opacity'],sigma_m=parameter_stats['sigma_m'],isolate_alpha_mean=float(A[support].mean()) if bool(support.any()) else None,inner_pixels=int(inner.sum()),static_occlusion_loss=float(1-W[inner].sum()/den),intrinsic_actor_mean_RGB=(cpu(iso['rgb'][:,inner].sum(1)/den)).tolist(),visible_actor_mean_RGB=cpu(color[:,inner].sum(1)/W[inner].sum().clamp_min(1e-8)).tolist(),requested_rotation_world=row['target_rotation_delta_world'],requested_center_world=edit['center'].tolist() if edit else f['actor_center_world'])
                    maps=dict(alpha=cpu(result['alpha'][0]).astype('float16'),dynamic_weight=cpu(W).astype('float16'),isolated_alpha=cpu(A).astype('float16'),visibility_ratio=cpu(ratio).astype('float16'),depth_m=cpu(result['depth_sum'][0]/result['alpha'][0].clamp_min(1e-8)/n.scale).astype('float16'))
                    np.savez_compressed(out/case/('%03d_maps.npz'%raw),**maps)
                    if raw in [98,101,109,113,114]:
                        save(out/case/'actor_rgb'/('%03d.png'%raw),iso['rgb'])
                        # Gray alpha and visibility are diagnostic maps, not RGB truth.
                        save(out/case/'isolated_alpha'/('%03d.png'%raw),A[None].repeat(3,1,1));save(out/case/'visible_weight'/('%03d.png'%raw),W[None].repeat(3,1,1));save(out/case/'visibility_ratio'/('%03d.png'%raw),ratio[None].repeat(3,1,1))
                    records.append(newrec)
                max_unchanged=max(max_unchanged,max(float((v-before[k]).abs().max()) if v.numel() else 0 for k,v in actor.state_dict().items()));update(phase='rendering',frames=status['frames']+1,last_case=case,last_raw=raw)
            print('BADCASE_RENDER',case,flush=True)
        # Causal row-removal probes: classify neither all target-mask rows nor box points as vehicles.
        agg=np.load(ROOT/'attribution_v2'/CTX/'aggregate.npz');m=agg['mass'][:ns];total=m[:,21];tf=m[:,20]/np.maximum(total,1e-8);vf=m[:,14:20].sum(1)/np.maximum(total,1e-8);hits=agg['target_support_frames'][:ns]
        appearance=(total>1)&(tf>.85)&(vf>.65)&(hits>=3);dep=np.load(ROOT/'depth_support_v1'/CTX/'aggregate.npz');depthok=dep['support_frames'][:ns]>=3;strict=appearance&depthok
        probes=[]
        for name,mask in [('depth_supported_only',strict),('appearance_affinity_only',appearance)]:
            mt=torch.from_numpy(mask).cuda();pr=[]
            for row in base_rows:
                raw=row['raw'];i=row['camera_index'];baseline=n.render(row,dict(family='delete'));changed=static_without(n,row,mt);diff=(changed['rgb']-baseline['rgb']).abs().mean(0);body=torch.from_numpy(np.load(dataset/('%03d.npz'%raw))['body']).cuda();pr.append(dict(raw=raw,target_region_change_MAE=float(diff[body].mean()),outside_target_change_MAE=float(diff[~body].mean()),outside_target_changed_pixels=int((diff[~body]>1/255).sum())))
                if raw in [98,101,109,113,114]:save(out/('static_probe_'+name)/'native_rgb'/('%03d.png'%raw),changed['rgb'])
            probes.append(dict(name=name,removed_rows=int(mask.sum()),records=pr,read_only_counterfactual=True,accepted_as_fix=False))
        assert max_unchanged==0 and n.unchanged()
        write(out/'summary.json',dict(context=CTX,parameters=parameter_stats,geometry=geometry,records=records,parameter_max_abs_change_during_all_edits=max_unchanged,opacity_scale_count_during_edit_unchanged=True,no_edit_V1_direct_bypass_float_max=bypass_max,delete_vs_saved_V42_native_byte_max=delete_byte_max,static_probes=probes,source_checkpoint_unchanged=True,new_training_updates=0,complete_background_fix=False,quality_accepted=False,human_verdict=None));update(phase='complete',wall_seconds=time.monotonic()-started)
if __name__=='__main__':main()
