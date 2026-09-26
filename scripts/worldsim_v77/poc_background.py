"""ProPainter 删除目标后，以六相机 RGB 重新运行冻结 Ω，保存每时刻背景点。"""
import argparse
import json
import pathlib
import sys
import time
import numpy as np
from PIL import Image

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from video_review import scene_frame
from evaluate import read_geometry,render

ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPLICIT-POC-20260926/r1')
P0=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-P0-24ACTOR-20260926/r1')
OLD=json.loads((P0/'registration.json').read_text())
SCENES={'scene_0230':list(range(0,50,5)),'scene_0255':list(range(0,100,10))}

def prepare():
    if (ROOT/'registration.json').exists():raise FileExistsError(ROOT/'registration.json')
    record={'task_id':'WS-V77-EXPLICIT-POC-20260926','run_dir':str(ROOT),
      'scenes':[],'frozen_omega_checkpoint':OLD['checkpoint'],'omega_source':OLD['omega_source'],
      'modules':['VGGT-Ω 512','SAM2.1 large GT-box prompted','ProPainter','Hunyuan3D-2.1'],
      'seed':7701,'training_steps':0,'human_verdict':None,
      'input_roles':'Ω 网络仅 RGB；SAM2 初始化和门控、米制 B 对齐、actor 放置使用 GT；背景尺度使用框外 LiDAR',
      'frame_timing':'processed 10Hz，原始 camera timestamp 不可得'}
    for name,frames in SCENES.items():
        spec=next(s for s in OLD['scenes'] if s['name']==name)
        root=ROOT/name; meta=json.loads((root/'selection.json').read_text())
        source=pathlib.Path(spec['root']);frame_rows=[]
        (root/'instances_snapshot.json').write_text((source/'instances/instances_info.json').read_text())
        for f in frames:
            directory=root/'bg_input'/f'{f:03d}';directory.mkdir(parents=True,exist_ok=True)
            camera_rows=[]
            for cam in range(6):
                mask=root/'masks'/f'cam{cam}'/f'{f:05d}.png'
                active=mask.is_file() and bool(np.asarray(Image.open(mask)).any())
                if active:
                    src=root/'inpaint'/f'cam{cam}'/'frames'/f'{f:04d}.png'
                    image=Image.open(src).convert('RGB')
                else:
                    src=source/'images'/f'{f:03d}_{cam}.jpg'
                    image=Image.open(src).convert('RGB').resize((688,384),Image.Resampling.BICUBIC)
                if image.size!=(688,384):raise AssertionError((name,f,cam,image.size))
                target=directory/f'cam{cam}.png';image.save(target)
                camera_rows.append({'camera':cam,'inpainted':active,'input':str(target),'source':str(src)})
            frame_rows.append({'frame':f,'cameras':camera_rows})
        record['scenes'].append({'name':name,'actor':meta['actor'],'root':str(source),'frames':frame_rows,
          'spec':spec,'role':'scene_0230 difficult' if name=='scene_0230' else 'cleaner visual control'})
    (ROOT/'registration.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'scenes':[(s['name'],len(s['frames'])) for s in record['scenes']]}))

def infer():
    import torch
    reg=json.loads((ROOT/'registration.json').read_text());torch.set_num_threads(4);torch.manual_seed(reg['seed'])
    assert torch.cuda.is_available()
    sys.path.insert(0,reg['omega_source'])
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera
    with torch.device('meta'):model=VGGTOmega()
    model.load_state_dict(torch.load(reg['frozen_omega_checkpoint'],map_location='cpu',weights_only=True,mmap=True),strict=True,assign=True)
    model.requires_grad_(False);model=model.eval().cuda()
    print('OMEGA_BACKGROUND_READY',flush=True)
    for s in reg['scenes']:
        instances=json.loads((ROOT/s['name']/'instances_snapshot.json').read_text())
        for entry in s['frames']:
            f=entry['frame'];out=ROOT/s['name']/'background'/f'{f:03d}';out.mkdir(parents=True,exist_ok=True)
            if (out/'summary.json').exists():continue
            scene=scene_frame(s['spec'],f,instances)
            for v,row in zip(scene['views'],entry['cameras']):v['image']=row['input']
            images=load_and_preprocess_images([v['image'] for v in scene['views']],image_resolution=512,mode='balanced').cuda()
            start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                raw=model(images);ex,k=encoding_to_camera(raw['pose_enc'],images.shape[-2:])
            torch.cuda.synchronize()
            pred={n:raw[n].float().cpu().numpy() for n in ['depth','depth_conf','pose_enc']}
            pred.update(extrinsics=ex.float().cpu().numpy(),intrinsics=k.float().cpu().numpy(),images=images.float().cpu().numpy())
            assert all(np.isfinite(v).all() for v in pred.values())
            np.savez_compressed(out/'prediction.npz',**pred)
            variants,cams,ks,lidar,original,alignment=read_geometry(scene,pred,out)
            points,colors,_=variants['calibrated_control']
            np.savez_compressed(out/'background_points.npz',points=points.astype(np.float32),colors=colors)
            for cam,(pose,kk) in enumerate(zip(cams,ks)):
                rgb,_=render(points,colors,pose,kk,(384,688))
                Image.fromarray(rgb).save(out/f'bg_cam{cam}.png')
            summary={'scene':s['name'],'frame':f,'bg_points':len(points),
              'inpainted_cameras':[r['camera'] for r in entry['cameras'] if r['inpainted']],
              'omega_forward_and_readout_s':time.monotonic()-start,
              'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
              'background_lidar_scale':alignment['calibrated_global_depth_scale'],
              'original_hw':list(original.shape[1:3]),'human_verdict':None}
            (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(summary,ensure_ascii=False),flush=True)
            del raw,images,pred,variants,points,colors,lidar,original;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','infer']);a=p.parse_args()
    globals()[a.command]()
