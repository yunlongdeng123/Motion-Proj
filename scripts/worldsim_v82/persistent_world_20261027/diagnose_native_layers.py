"""Read-only asset/occluder isolation, not a repair method or edited output."""
import argparse,json,os,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_erosion
from gpu_slot import exclusive
from resource_control import require_start
from render_motion_suite import ROOT,NativeMotion,prepare_rasterization,helpers,torch,cpu,actual_native_render


def png(path,value):
    Image.fromarray(np.round(np.clip(value,0,1)*255).astype('uint8')).save(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--context',required=True);p.add_argument('--output',type=Path);a=p.parse_args();require_start()
    run=ROOT/'native_scene_training_v1'/a.context/'formal30k'
    state_path=ROOT/'motion_suite_execution_v1'/a.context/'status.json';state=json.loads(state_path.read_text())
    reference=Path(state.get('native_suite_path',str(state_path.parent/'native_suite')))
    contract=ROOT/'motion_edit_suite_v1'/a.context
    assert json.loads((reference/'summary.json').read_text())['frames']==31
    dest=a.output or ROOT/'native_layer_isolation_v1'/a.context;dest.mkdir(parents=True,exist_ok=False)
    status=dict(context=a.context,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',new_training_updates=0,new_edit_case_count=0)
    def update(**kw):
        status.update(kw);tmp=dest/'status.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(dest/'status.json')
    update()
    try:
        with exclusive(),torch.no_grad():
            require_start();update(phase='GPU_layer_isolation');started=time.monotonic();torch.cuda.reset_peak_memory_stats()
            native=NativeMotion(run,30000);cases=['no_edit','translate_lateral_1m','turn_left15','turn_right15']
            records=[];panels={};maximum_replay=0.;maximum_weight_difference=0.;maximum_visibility_violation=0.
            for name in cases:
                case=json.loads((contract/(name+'.json')).read_text());folder=dest/name;folder.mkdir()
                for index,row in enumerate(case['frames']):
                    require_start();stem=f'{row["raw"]:03d}';camera=native.camera(row,case);query=native.camera(row,case,query=True)
                    kwargs=native.dynamic_kwargs(query,row,case,True)
                    z=torch.cat((kwargs['means3D'],torch.ones_like(kwargs['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3]
                    kwargs['features']=torch.cat((z,torch.ones_like(z),torch.ones_like(z)),1)
                    isolated=helpers.raster(camera,native.args,native.static.active_sh_degree,kwargs,native.bg)
                    color=cpu(isolated['rgb']).transpose(1,2,0);alpha=cpu(isolated['alpha'])[0]
                    assert np.isfinite(color).all() and np.isfinite(alpha).all()
                    weight_difference=float(np.abs(cpu(isolated['extra'])[0]-alpha).max());maximum_weight_difference=max(maximum_weight_difference,weight_difference);assert weight_difference<2e-5
                    with np.load(reference/name/'float'/(stem+'.npz')) as combined:
                        weight=combined['dynamic_weight'][0];joint_rgb=combined['rgb'].transpose(1,2,0).copy()
                        assert np.isfinite(weight).all()
                    violation=float(np.max(weight-alpha));maximum_visibility_violation=max(maximum_visibility_violation,violation)
                    assert violation<.002,'Dynamic-only alpha and mixed visibility must be consistent within raster early-termination precision'
                    core=binary_erosion(alpha>.1,iterations=2);denominator=int(core.sum());mass=float(alpha[core].sum())
                    record=dict(case=name,raw=row['raw'],dynamic_support_pixels=int((alpha>.1).sum()),dynamic_core_pixels=denominator,
                                isolated_dynamic_alpha_mean=float(alpha[core].mean()) if denominator else None,
                                mixed_dynamic_weight_mean=float(weight[core].mean()) if denominator else None,
                                static_visibility_suppressed_alpha_mass=float((alpha[core]-weight[core]).sum())/mass if mass else None,
                                mixed_dynamic_weight_below_half_isolated_fraction=float((weight[core]<.5*alpha[core]).mean()) if denominator else None,
                                maximum_visibility_inequality_violation=violation)
                    records.append(record)
                    if index in [0,15,23,30]:
                        replay=native.render(row,case);delta=float(np.max(np.abs(cpu(replay['rgb']).transpose(1,2,0)-joint_rgb)));maximum_replay=max(maximum_replay,delta);assert delta<=2e-6
                        if name=='no_edit':
                            actual=actual_native_render(camera,native.static,native.dynamic,native.args,native.bg,env_map=native.env)['render']
                            assert float((actual-replay['rgb']).abs().max())<=2e-6
                        with np.load(reference/'delete_target/float'/(stem+'.npz')) as static:static_rgb=static['rgb'].transpose(1,2,0).copy()
                        neutral=color+(1-alpha[...,None])*.5
                        png(folder/(stem+'_dynamic_on_gray.png'),neutral);png(folder/(stem+'_dynamic_alpha.png'),np.repeat(alpha[...,None],3,axis=2))
                        np.savez_compressed(folder/(stem+'_isolated_float.npz'),rgb_premultiplied=color,alpha=alpha,mixed_dynamic_weight=weight)
                        if index in [23,30]:
                            support=alpha>.05
                            body=np.array(Image.open(native.source/'dynamic_masks/dynamic_mask_0'/(row['frame_id']+'.png')).resize((480,320),Image.Resampling.NEAREST))>0
                            support|=body;yy,xx=np.where(support)
                            bounds=[max(0,int(xx.min())-15),max(0,int(yy.min())-15),min(480,int(xx.max())+16),min(320,int(yy.max())+16)] if len(xx) else [0,0,480,320]
                            panels[(index,name)]=(joint_rgb,neutral,np.repeat(alpha[...,None],3,axis=2),static_rgb,bounds,stem)
            assert native.unchanged()
            peaks=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            seconds=time.monotonic()-started
        for index in [23,30]:
            sheet=Image.new('RGB',(1920,1400),'#142332');draw=ImageDraw.Draw(sheet)
            for row,name in enumerate(cases):
                *images,bounds,stem=panels[(index,name)]
                for column,(label,value) in enumerate(zip(['native composite','dynamic only over gray','dynamic only alpha','static only + environment'],images)):
                    image=Image.fromarray(np.round(np.clip(value,0,1)*255).astype('uint8')).crop(tuple(bounds)).resize((480,320),Image.Resampling.NEAREST)
                    sheet.paste(image,(column*480,row*350+30));draw.text((column*480+4,row*350+8),name+' / '+label+' / raw'+stem,fill='white')
            sheet.save(dest/f'comparison_index{index}.jpg',quality=95)
        result=dict(context=a.context,frames=31,cases=cases,rows=records,actual_native_and_saved_composite_replay_max_difference=maximum_replay,
                    isolated_dynamic_weight_vs_alpha_max_difference=maximum_weight_difference,visibility_inequality_violation_max=maximum_visibility_violation,
                    master_unchanged=True,GPU_seconds=seconds,cuda_allocator_peak=peaks,new_training_updates=0,new_edit_case_count=0,
                    diagnostic_only=True,static_layer_not_edited=True,opacity_visibility_is_not_semantic_contamination=True,
                    depth_not_used_as_first_hit_GT=True,counterfactual_RGB_GT=False,quality_accepted=False,human_verdict=None)
        (dest/'summary.json').write_text(json.dumps(result,indent=2));update(phase='complete');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    except Exception:
        update(phase='engineering_failure_preserved',error=traceback.format_exc());raise

if __name__=='__main__':main()
