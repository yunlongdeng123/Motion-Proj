"""真实连续曝光的单次长窗工程探针；反向只验梯度，优化零步。"""
from pathlib import Path
import argparse, os, sys, time, json, traceback
os.environ['DRIVEEDITOR_SEQUENTIAL_CFG']='1'
os.environ['OMP_NUM_THREADS']='4'
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
S=P/'target_protected';T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
sys.path[:0]=[str(P),str(S),str(S/'iteration7'),str(S/'iteration11')]
import numpy as np
import torch
from PIL import Image


def main(a):
    torch.set_num_threads(4)
    root=T/'r21';out=root/f'{a.phase}_{a.frames}';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('已保存探针结果，不重复执行')
    state={'phase':a.phase,'frames':a.frames,'size':[320,576],'optimizer_steps':0,'pid':os.getpid(),'stage':'loading'}
    def save(): (out/'state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    save();start=time.monotonic()
    folder=T/'r16/synthetic/L001'
    load=lambda role,rgb:np.stack([np.asarray(Image.open(p).convert('RGB' if rgb else 'L')) for p in sorted((folder/role).glob('*.png'))[:a.frames]])
    y,x,h=load('Y',True),load('X',True),load('model_hole',False)>0
    assert len(y)==a.frames
    from train_control import resized
    prepared=resized(y,x,h,[320,576])
    timestamps=[f['timestamp'] for f in json.loads((folder/'pair_manifest.json').read_text())['frames'][:a.frames]]
    assert all(t2>t1 for t1,t2 in zip(timestamps,timestamps[1:]))
    state.update(source_frames_unique=a.frames,exposure_span_s=(timestamps[-1]-timestamps[0])/1e6)
    try:
        if a.phase=='train':
            import train_control as trainer
            from temporal_scope import selected
            trainer.selected=selected;trainer.MODULES='temporal_self'
            trainer.ENCODER=T/'r7/encoder_recovery/official_svd_encoder.safetensors'
            model,params,names=trainer.model_init();calls=[]
            def capture(module,args,kwargs):
                calls.append({'frames':int(kwargs['num_video_frames']),'latent_shape':list(args[0].shape) if args else None})
            handle=model.model.diffusion_model.register_forward_pre_hook(capture,with_kwargs=True)
            torch.cuda.reset_peak_memory_stats();state['stage']='forward_backward';save()
            loss=trainer.loss(model,prepared,6201);assert torch.isfinite(loss)
            loss.backward()
            finite=sum(p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().max().item()>0 for p in params)
            assert finite==80,(finite,len(params))
            assert calls and all(c['frames']==a.frames for c in calls)
            handle.remove();state.update(loss=float(loss.detach()),finite_nonzero_gradient_tensors=finite,network_calls=calls)
        else:
            from repair_drive import Engine,set_seed
            e=Engine(num_frames=a.frames,out_size=(320,576),num_steps=3)
            e.masked_condition=prepared[1]
            e.im=[im for im in ((prepared[0].permute(0,2,3,1)+1)*127.5).round().clamp(0,255).byte().numpy()]
            e.masks=[im for im in prepared[2].numpy()];calls=[]
            def capture(module,args,kwargs):calls.append(int(kwargs['num_video_frames']))
            handle=e.model.model.diffusion_model.register_forward_pre_hook(capture,with_kwargs=True)
            torch.cuda.reset_peak_memory_stats();set_seed(42);state['stage']='sampling_three_steps';save();e.predict(1,False,'Deletion')
            assert len(e.im_result)==a.frames and calls and all(c==a.frames for c in calls)
            for i,im in enumerate(e.im_result):
                assert im.shape==(320,576,3) and im.dtype==np.uint8
                Image.fromarray(im).save(out/f'{i:05}.png')
            handle.remove();state.update(network_call_frame_counts=calls,output_frames=len(e.im_result),sampler_steps=3,quality_claim=False)
        state.update(stage='passed',peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,seconds=time.monotonic()-start)
    except Exception as ex:
        state.update(stage='OOM' if isinstance(ex,torch.cuda.OutOfMemoryError) else 'engineering_error',error=repr(ex),traceback=traceback.format_exc(),seconds=time.monotonic()-start)
        (out/'result.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');save();print(json.dumps(state,ensure_ascii=False),flush=True);raise
    (out/'result.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');save();print(json.dumps(state,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['train','infer'],required=True);p.add_argument('--frames',type=int,choices=[20,30],required=True);main(p.parse_args())
