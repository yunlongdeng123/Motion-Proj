"""冻结原生渲染：source-only 局部 unknown 控制臂；每帧独立，无训练。"""
import sys,json,time,argparse,gc,os
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
sys.path.insert(0,str(ROOT/'unknown_detection_v3/code'))
from probe_detection import attribution,kwargs,torch,cpu,write,save,exclusive
from train_actor import Actor,NativeMotion,OLD,helpers
from scope import build_scope,build_query,eligible_from_screen,compose_bounded_white,selection_start

def auth(ctx):
    a=json.loads((ROOT/'authorization.json').read_text())
    assert a.get('allow_temporal_multiscene') and ctx in a.get('temporal_contexts',[])
    assert not a.get('gpu_execution_hold') and not a.get('user_research_hold') and not (ROOT/'USER_STOP').exists()

def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raws',default='all');p.add_argument('--arms',default='source_body_contact,source_local_band');a=p.parse_args()
    auth(a.context);a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();records=[]
    write(a.output/'status.json',dict(phase='waiting_gpu',pid=os.getpid(),training_updates=0))
    with exclusive():
        torch.set_num_threads(1);ds=ROOT/'datasets'/a.context;frames=json.loads((ds/'manifest.json').read_text())['frames']
        ck=ROOT/'actor_campaign_v1'/a.context/'formal10k/checkpoint010000.pth'
        actor=Actor(np.load(ds/'seeds.npz'));actor.load_state_dict(torch.load(str(ck),map_location='cuda')['actor']);actor.requires_grad_(False)
        n=NativeMotion(OLD/'native_scene_training_v1'/a.context/'formal30k',30000);ns=len(n.static._xyz)
        namespace=str(OLD/'native_scene_training_v1'/a.context/'formal30k')+'@30000:static'
        wanted=None if a.raws=='all' else set(map(int,a.raws.split(',')))
        for i,f in enumerate(frames):
            raw=f['raw']
            if wanted is not None and raw not in wanted:continue
            auth(a.context);ob=np.load(ds/f'{raw:03d}.npz');scope=build_scope(ob['body'],context=a.context,raw=raw,frame_id=f.get('frame_id'),body_source=str(ds/f'{raw:03d}.npz'))
            if scope['status']!='ok':
                records.append(dict(raw=raw,status='skipped',reason=scope['skip_reason']));continue
            cam,st,dy=kwargs(n,actor,f,i)
            with torch.no_grad():
                baseout=helpers.raster(cam,n.args,n.static.active_sh_degree,st,n.bg)
                env=cpu(n.env(cam.get_world_directions(False).permute(1,2,0)).permute(2,0,1))
                basealpha=cpu(baseout['alpha'][0]);baseg=cpu(baseout['rgb'])-(1-basealpha)[None]*cpu(n.bg).reshape(3,1,1)
                baseline=baseg+(1-basealpha)[None]*env
                full=helpers.raster(cam,n.args,n.static.active_sh_degree,helpers.merge_rows(st,dy),n.bg)
                fullrgb=cpu(full['rgb'])+(1-cpu(full['alpha']))*env
                h=torch.cat((st['means3D'],torch.ones_like(st['means3D'][:,:1])),1);ndc=h@cam.full_proj_transform;ndc=ndc[:,:3]/ndc[:,3:4]
                screen=cpu(torch.stack(((ndc[:,0]+1)*cam.image_width/2-.5,(ndc[:,1]+1)*cam.image_height/2-.5),1))
                z=cpu((h@cam.world_view_transform)[:,2]);eligible=eligible_from_screen(screen,z,scope['allowed'])
            del baseout,full
            for arm in a.arms.split(','):
                q=build_query(scope,arm);query=q['query'];dest=a.output/arm/f'{raw:03d}';dest.mkdir(parents=True)
                maps=np.stack([query,scope['user_contact_query'],~scope['allowed'],np.ones_like(query)]).astype('float32')
                initial=selection_start('empty');removed=np.zeros(ns,bool);history=[];converged=False
                for step in range(9):
                    kw={**st,'opacities':st['opacities'].clone()};kw['opacities'][torch.from_numpy(removed).cuda()]=0
                    out,w,c=attribution(n,cam,kw,maps);w=cpu(w);alpha=cpu(out['alpha'][0]);g=cpu(out['rgb'])-(1-alpha)[None]*cpu(n.bg).reshape(3,1,1)
                    add=(w[:,0]>1e-7)&eligible&~removed
                    history.append(dict(step=step,new_rows=int(add.sum()),query_alpha_mean=float(alpha[query].mean()),adjoint_closure=c,outside_center_remaining_rows=int(((w[:,0]>1e-7)&~eligible&~removed).sum())))
                    if not add.any():converged=True;break
                    if step==8:break
                    removed|=add;del out;gc.collect()
                output=compose_bounded_white(scope,g,alpha,baseline);ungated=g+(1-alpha)[None];native=g+(1-alpha)[None]*env
                dsco=scope['display_scope'];bounded_native=baseline.copy();bounded_native[:,dsco]=native[:,dsco]
                assert not (removed&~eligible).any();assert np.array_equal(output[:,~dsco],baseline[:,~dsco])
                for name,x in [('white',output),('native',bounded_native),('ungated_white',ungated),('original',fullrgb),('actor_omitted',baseline),('gaussian',g)]:save(dest/(name+'.png'),torch.from_numpy(x))
                Image.fromarray(np.round(np.clip(alpha,0,1)*255).astype('uint8')).save(dest/'alpha.png')
                Image.fromarray(np.round(np.clip(ob['rgb'],0,255)).astype('uint8')).save(dest/'observed.png')
                remaining=(w[:,0]>1e-7)&~removed
                np.savez_compressed(dest/'evidence.npz',starting_removed_ids=initial,existing_removed_ids=initial,final_removed_ids=np.flatnonzero(removed).astype('int32'),new_unknown_ids=np.flatnonzero(removed).astype('int32'),eligible_ids=np.flatnonzero(eligible).astype('int32'),screen_centers=screen[removed].astype('float32'),allowed=scope['allowed'],query=query,display_scope=dsco,body=scope['body'],contact=scope['user_contact_query'],alpha=alpha,gaussian_rgb=g,white_rgb=output,native_rgb=bounded_native,baseline_native_rgb=baseline,environment_rgb=env,residual_ids=np.flatnonzero(remaining).astype('int32'),residual_mass=w[remaining,0],row_namespace=np.array(namespace),checkpoint_ref=np.array(str(ck)),context=np.array(a.context),raw=np.array(raw))
                lum=(np.clip(output,0,1)*np.array([.2126,.7152,.0722])[:,None,None]).sum(0)
                rec=dict(context=a.context,raw=raw,frame_id=f.get('frame_id',n.records[i]['img_name']),camera_index=i,arm=arm,status='ok',method_arm=arm,query_mode=arm,init_mode='empty',native_RGB_content_used=False,static_rows=ns,template=scope['template'],method_parameters=dict(mass_epsilon=1e-7,max_iterations=8),model_time=f['model_time'],edit_case='delete_target',history=history,converged=converged,initialization='empty',starting_removed_rows=0,existing_removed_rows=0,independent_frame_reset=True,row_namespace=namespace,checkpoint_ref=str(ck),final_removed_rows=int(removed.sum()),new_rows_outside_blue_centers=int((removed&~eligible).sum()),outside_output_max_difference=float(np.abs(output[:,~dsco]-baseline[:,~dsco]).max()),body_bbox_xyxy=scope['body_bbox_xyxy'],query_pixels=int(query.sum()),display_pixels=int(dsco.sum()),contact_dark_pixels=int(((lum<.2)&scope['user_contact_query']).sum()),contact_alpha_mean=float(alpha[scope['user_contact_query']].mean()),query_alpha_mean=float(alpha[query].mean()),remaining_outside_center_rows=int((remaining&~eligible).sum()),final_after_recomputed=True,after_attribution_recomputed=True,actual_rasterizer_rerendered=True,training_updates=0,labels_for_training=False,outside_baseline='same-frame actor-omitted static+environment',white_equation='G_after+(1-alpha_after)*white inside local display band; baseline outside',query_provenance=q['query_provenance'],semantic_unknown_claim=False,human_verdict=None)
                write(dest/'receipt.json',rec);records.append(rec);print('FRAME',a.context,arm,raw,int(removed.sum()),converged,flush=True)
                del out,w,kw;gc.collect()
            del st,dy;gc.collect()
            write(a.output/'status.json',dict(phase='rendering',pid=os.getpid(),last_raw=raw,completed=len(records),wall_seconds=time.monotonic()-start))
        assert n.unchanged()
    summary=dict(context=a.context,records=records,source_checkpoint_unchanged=True,training_updates=0,wall_seconds=time.monotonic()-start,cuda_peak_reserved=torch.cuda.max_memory_reserved(),scope='same frozen normalized user blue polygon, transported by observed source body bbox',not_same_initialization_as_accepted_R2=True)
    write(a.output/'summary.json',summary);write(a.output/'status.json',dict(phase='complete',completed=len(records),wall_seconds=time.monotonic()-start))
if __name__=='__main__':main()
