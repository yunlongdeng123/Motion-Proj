"""固定预训练DiffuEraser，显式接收原生三维prior；PNG I/O保留源帧。"""
import argparse,json,math,os,sys,time,types
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from PIL import Image
SOURCE=Path('/root/autodl-tmp/third_party/worldsim_v77/DiffuEraser')
WEIGHTS=Path('/root/autodl-tmp/models/persistent_world_diffueraser_20261027')
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
sys.path.insert(0,str(SOURCE))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cases',nargs='+',default=['delete_all_unrepaired','delete_all_repaired','delete_all_evidence_routed','delete_all_direct_reference']);a=parser.parse_args()
    assert json.loads((ROOT/'download_status.json').read_text())['status']=='complete'
    runtime=ROOT/'diffusion_runtime';runtime.mkdir(exist_ok=True)
    if not (runtime/'weights').exists():(runtime/'weights').symlink_to(WEIGHTS,target_is_directory=True)
    assert (runtime/'weights').resolve()==WEIGHTS
    os.chdir(runtime);torch.set_num_threads(2);torch.set_grad_enabled(False)
    current={}
    def frames(path):return [Image.open(p).convert('RGB') for p in sorted(Path(path).glob('*.png'))]
    def video(path,seconds,nframes,max_side):
        seq=frames(path);assert len(seq)==49
        return seq,10.,seq[0].size,math.ceil(len(seq)/nframes),len(seq)
    def masks(path,fps,count,size,dilation,inputs):
        assert fps==10 and count==49 and dilation==0
        seq=[Image.open(p).convert('L').resize(size,Image.Resampling.NEAREST) for p in sorted(Path(path).glob('*.png'))]
        assert len(seq)==count
        images=[Image.fromarray(np.where((np.asarray(m)>0)[...,None],0,np.asarray(f)).astype('uint8')) for f,m in zip(inputs,seq)]
        return seq,images
    def capture(seq,label):
        path=current['output']/label;path.mkdir(exist_ok=True)
        assert len(seq)>=49
        for original,image in zip(current['frames'],seq):Image.fromarray(np.asarray(image,dtype=np.uint8)).save(path/original.name)
    code=(SOURCE/'diffueraser/diffueraser.py').read_text()
    for old,new in [('        ################ Compose ################','        capture(images,"model_png")\n        ################ Compose ################'),
                    ('        default_fps = fps','        capture(comp_frames,"final_png")\n        default_fps = fps')]:
        assert code.count(old)==1;code=code.replace(old,new)
    module=types.ModuleType('persistent_diffueraser_io');module.__file__=str(SOURCE/'diffueraser/diffueraser.py');module.capture=capture
    exec(compile(code,module.__file__,'exec'),module.__dict__)
    module.read_video=video;module.read_mask=masks
    module.read_priori=lambda path,fps,n,size:[image.resize(size,Image.Resampling.BILINEAR) for image in frames(path)]
    started=time.time();model=module.DiffuEraser(torch.device('cuda'),str(WEIGHTS/'stable-diffusion-v1-5'),str(WEIGHTS/'sd-vae-ft-mse'),str(WEIGHTS/'diffuEraser'),ckpt='2-Step')
    init=dict(seconds=time.time()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
    (ROOT/'diffusion_model_load.json').write_text(json.dumps(init,indent=2))
    for case in a.cases:
        src=ROOT/'phase1_native'/case;dest=src/'diffueraser';dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():continue
        current.update(output=dest,frames=sorted((src/'native_rgb').glob('*.png')))
        status=dict(phase=case,pid=os.getpid(),started=time.time(),backend='DiffuEraser official PCM2',prior='native upstream RGB, not ProPainter output')
        (ROOT/'diffusion_status.json').write_text(json.dumps(status,indent=2))
        torch.manual_seed(0);np.random.seed(0);torch.cuda.reset_peak_memory_stats();t=time.time()
        model.forward(str(src/'native_rgb'),str(src/'mask'),str(src/'native_rgb'),str(dest/'video.mp4'),
            max_img_size=480,video_length=4.9,mask_dilation_iter=0,nframes=22,seed=0,guidance_scale=0,blended=False)
        rows=[]
        for original in current['frames']:
            before=np.asarray(Image.open(original));mask=np.asarray(Image.open(src/'mask'/original.name))>0
            generated=np.asarray(Image.open(dest/'final_png'/original.name))
            error=int(np.abs(generated.astype(float)-before)[~mask].max());assert error==0
            rows.append(dict(raw=int(original.stem),generation_pixels=int(mask.sum()),outside_max_difference=error))
        receipt=dict(status='complete',case=case,frames=49,seconds=time.time()-t,rows=rows,model_initialization=init,
            seed=0,steps=2,nframes=22,mask_dilation=0,resolution=[480,320],fps=10,
            native_prior=True,training_updates=0,generated_is_3d_state=False,
            cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),quality_accepted=False,human_verdict=None)
        (dest/'summary.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
    (ROOT/'diffusion_status.json').write_text(json.dumps(dict(phase='video_comparison_report_pending',inference_complete=True,cases=a.cases),indent=2))


if __name__=='__main__':
    from gpu_slot import exclusive
    with exclusive():main()
