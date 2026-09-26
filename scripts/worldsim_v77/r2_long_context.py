"""同ProPainter权重的全196帧上下文控制；冻结原评价窗mask与评价帧。"""
import argparse,json,os,pathlib,subprocess,sys,time
import numpy as np
from PIL import Image
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from geometry import resized_intrinsics,project_bbox
OLD=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
OUT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R2-20260926/r1')
SOURCE=pathlib.Path('/root/autodl-tmp/third_party/worldsim_v77/ProPainter-hfspace')
def prepare(name,actor,cam,old_count):
    root=OUT/name;data=pathlib.Path('/root/autodl-tmp/data/v76_vadgs')/name
    fa=json.loads((data/'instances/instances_info.json').read_text())[actor]['frame_annotations']
    files=sorted((data/'images').glob(f'*_{cam}.jpg'));count=len(files);assert count==196
    vals=np.loadtxt(data/'intrinsics'/f'{cam}.txt');K=resized_intrinsics([[vals[0],0,vals[2]],[0,vals[1],vals[3]],[0,0,1]],(1600,900),(384,688))
    rgb=root/'rgb'/f'cam{cam}';masks=root/'masks'/f'cam{cam}';rgb.mkdir(parents=True,exist_ok=True);masks.mkdir(parents=True,exist_ok=True)
    boxes=[]
    for f in range(count):
        b=None
        if f in fa['frame_idx']:
            j=fa['frame_idx'].index(f);b=project_bbox(np.array(fa['obj_to_world'][j]),fa['box_size'][j],np.loadtxt(data/'extrinsics'/f'{f:03d}_{cam}.txt'),K,(384,688))
            if b and (b[2]-b[0])*(b[3]-b[1])<400:b=None
        boxes.append(b)
        target=rgb/f'{f:05d}.jpg'
        if not target.exists():
            if f<old_count:target.symlink_to(OLD/name/'rgb'/f'cam{cam}'/target.name)
            else:Image.open(data/'images'/f'{f:03d}_{cam}.jpg').convert('RGB').resize((688,384),Image.Resampling.BICUBIC).save(target,quality=95)
        mask=masks/f'{f:05d}.png'
        if not mask.exists() and f<old_count:mask.symlink_to(OLD/name/'masks'/f'cam{cam}'/mask.name)
        if not mask.exists() and b is None:Image.new('L',(688,384)).save(mask)
    metadata={'scene':name,'actor_id':actor,'camera':cam,'frames':count,'old_input_frames':old_count,'new_visible_frames':[f for f,b in enumerate(boxes) if b is not None and f>=old_count],'all_active_frames':[f for f,b in enumerate(boxes) if b is not None],'original_window_masks_frozen':True,'boxes':boxes,'human_verdict':None}
    (root/'context_registration.json').write_text(json.dumps(metadata,indent=2)+'\n');return metadata
def masks(meta):
    import torch
    torch.set_num_threads(4);torch.manual_seed(7703)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
    from sam2.build_sam import build_sam2_video_predictor
    root=OUT/meta['scene'];c=meta['camera'];folder=root/'masks'/f'cam{c}'
    if all((folder/f'{f:05d}.png').exists() for f in range(meta['frames'])):return
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda');predictor.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=predictor.init_state(video_path=str(root/'rgb'/f'cam{c}'),offload_video_to_cpu=True,offload_state_to_cpu=True)
        first=meta['all_active_frames'][0];predictor.add_new_points_or_box(state,frame_idx=first,obj_id=1,box=np.array(meta['boxes'][first],np.float32))
        for f,_,logits in predictor.propagate_in_video(state,start_frame_idx=first,max_frame_num_to_track=meta['frames']-first):
            target=folder/f'{f:05d}.png'
            if target.exists():continue
            raw=(logits[0,0]>0).cpu().numpy().astype(np.uint8);b=meta['boxes'][f]
            if b is None:raw[:]=0
            else:
                x0,y0,x1,y1=[int(round(v)) for v in b];gate=np.zeros_like(raw);gate[max(0,y0-6):min(384,y1+7),max(0,x0-6):min(688,x1+7)]=1;raw &= gate
            Image.fromarray(raw*255).save(target)
    assert all((folder/f'{f:05d}.png').exists() for f in range(meta['frames']))
def paint(meta):
    root=OUT/meta['scene'];c=meta['camera'];summary=root/'inpaint'/f'cam{c}'/'summary.json'
    if summary.exists():return
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    cmd=[sys.executable,str(SOURCE/'inference_propainter.py'),'--video',str(root/'rgb'/f'cam{c}'),'--mask',str(root/'masks'/f'cam{c}'),'--output',str(root/'inpaint'),'--width','688','--height','384','--save_fps','10','--save_frames','--fp16','--subvideo_length','200','--neighbor_length','10','--ref_stride','10']
    start=time.monotonic();print('PROPAINTER_LONG_CONTEXT',json.dumps(cmd),flush=True)
    with (root/'propainter_long_context.log').open('w') as log:subprocess.run(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    paths=sorted((summary.parent/'frames').glob('*.png'));assert len(paths)==meta['frames']
    summary.write_text(json.dumps({'scene':meta['scene'],'camera':c,'frames':len(paths),'elapsed_s':time.monotonic()-start,'command':cmd,'training_steps':0,'human_verdict':None},indent=2)+'\n')
    print('LONG_CONTEXT_DONE',meta['scene'],len(paths),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','masks','paint']);p.add_argument('--scene',choices=['scene_0230','scene_0255'],required=True);a=p.parse_args()
    actor,cam,n={'scene_0230':('22',2,50),'scene_0255':('25',3,100)}[a.scene]
    if a.command=='prepare':print(json.dumps(prepare(a.scene,actor,cam,n)),flush=True)
    else:
        meta=json.loads((OUT/a.scene/'context_registration.json').read_text());globals()[a.command](meta)
