"""No-update GN11/native-renderer -> old3 -> old5 static RGB provenance."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
TASK=ROOT/'identity_completion_20261018';OUT=ROOT/'baseline_engineering_20261018'
sys.path.insert(0,str(TASK/'infra'))
from priority import require_gpu_slot
from runtime import load_models,masks_for_frame

def main():
    slot,admission=require_gpu_slot(str(ROOT.parent/'gpu_slot.lock'))
    out=OUT/'prior_trace'
    if out.exists():raise FileExistsError('preserve trace output')
    out.mkdir();start=time.monotonic();r=dict(admission=admission,parameter_updates=0)
    try:
        import torch,numpy as np
        from PIL import Image
        from renderer import render
        torch.set_num_threads(2);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        spec=json.loads((TASK/'spec5.json').read_text());scratch=out/'loader';scratch.mkdir()
        cfg,scene,s,m,opt,env,exclusion=load_models(spec,scratch,freeze_background=False)
        from gaussian_renderer import render as native_render
        frames=[]
        def png(p,t):
            arr=np.rint(t.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            Image.fromarray(arr).save(p);return arr
        with torch.no_grad():
            for i,cam in enumerate(scene.getTrainCameras()):
                fid=Path(cam.image_name).stem;f=out/fid;f.mkdir()
                other=[s.get_scaling_t.clamp_max(2),s.get_inst_velocity]
                a=render(cam,s,None,cfg,torch.zeros(3,device='cuda'),env_map=env,other=other,render_separate=False)
                b=native_render(cam,s,None,cfg,torch.zeros(3,device='cuda'),env_map=env,other=other)
                diff=(a['render']-b['render']).abs()
                row=dict(frame_id=fid,native_adapter_max_abs=float(diff.max()),native_adapter_mae=float(diff.mean()))
                png(f/'GN11_static.png',b['render']);png(f/'observed.png',cam.original_image.cuda())
                for model in ['old3','old5']:
                    path=TASK/('renders_'+model)/fid/'static.png'
                    assert path.is_file(),str(path)
                    Image.open(path).save(f/(model+'_static.png'))
                frames.append(row)
                (OUT/'prior_trace_progress.json').write_text(json.dumps(dict(frames=i+1,total=49)))
        r.update(status='PASSED_NATIVE_ADAPTER_RGB' if max(x['native_adapter_max_abs'] for x in frames)<2e-4 else 'FAILED_NATIVE_ADAPTER_RGB',
            frames=frames,source_background=spec['background_checkpoint'],source_environment=spec['environment_checkpoint'],
            native_renderer_sha256=hashlib.sha256(Path(sys.modules['gaussian_renderer'].__file__).read_bytes()).hexdigest(),
            body_mask_or_inpainting_applied=False,static_points=len(s.get_xyz),
            scope='Same-view static provenance only; occluded road has no clean target and is not assigned a quality pass')
    except BaseException:
        r.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        r['seconds']=time.monotonic()-start;(OUT/'prior_trace_receipt.json').write_text(json.dumps(r,indent=2));slot.close()
        print(json.dumps(r),flush=True)

if __name__=='__main__':main()
