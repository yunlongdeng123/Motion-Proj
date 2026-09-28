"""同一SAM规则重跑九例全部目标视角；空mask保留为实际失败。"""
from nine_common import *
import os,time,numpy as np,torch,cv2
from PIL import Image,ImageDraw

reg=read(ROOT/'registration.json');torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(42)
assert not (ROOT/'mask_state.json').exists()
state=dict(state='running',pid=os.getpid(),completed=[],human_verdict=None);dump(ROOT/'mask_state.json',state)
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2_video_predictor
predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda')
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    for s in reg['scenes']:
        base=ROOT/s['name'];frames=read(base/'camera_frames.json');asset_candidates=[]
        for v in s['streams']:
            c=v['camera'];dest=base/f'cam{c}';(dest/'rgb').mkdir(parents=True,exist_ok=False)
            for f in range(30):
                im=Image.open(Path(s['root'])/'images'/f'{f:03}_{c}.jpg').convert('RGB').resize((1024,576),Image.Resampling.BILINEAR)
                im.save(dest/'rgb'/f'{f:05}.png')
                if v['active']:im.save(dest/'rgb'/f'{f:05}.jpg',quality=97)
            if not v['active']:continue
            started=time.time();raw={};track=predictor.init_state(video_path=str(dest/'rgb'),offload_video_to_cpu=True,offload_state_to_cpu=True);j=v['prompt_frame']
            predictor.add_new_points_or_box(track,frame_idx=j,obj_id=1,box=np.array(v['boxes'][j],np.float32))
            for reverse in [False,True]:
                for f,_,logit in predictor.propagate_in_video(track,start_frame_idx=j,reverse=reverse):raw[f]=(logit[0,0]>0).cpu().numpy()
            assert len(raw)==30
            for folder in ['sam','core','write_mask','model_mask','protect','alpha']:(dest/folder).mkdir()
            stats=[];sheet=Image.new('RGB',(1536,322*3),(20,30,40));draw=ImageDraw.Draw(sheet)
            for f,fr in enumerate(frames):
                c2w,k=camera(fr,c);actor=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);gate=hull_mask(actor,c2w,k,pad=3)
                core=largest(raw[f]&gate);write=cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0;write &= gate
                model=np.zeros((576,1024),bool)
                if write.any():
                    yy,xx=np.where(write);model[max(0,yy.min()-8):min(576,yy.max()+25),max(0,xx.min()-8):min(1024,xx.max()+9)]=True
                protect=np.zeros_like(model)
                for other in fr['all_boxes']:
                    if other['actor_id']!=s['actor']:protect |= hull_mask(other,c2w,k)
                protect &= model&~write
                dist=cv2.distanceTransform(model.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);t=np.clip(dist/8,0,1);alpha=t*t*(3-2*t);alpha[write]=1;alpha[protect]=0
                for folder,arr in [('sam',raw[f]*255),('core',core*255),('write_mask',write*255),('model_mask',model*255),('protect',protect*255),('alpha',np.rint(alpha*255))]:Image.fromarray(arr.astype('uint8')).save(dest/folder/f'{f:05}.png')
                stats.append(dict(frame=f,raw=int(raw[f].sum()),core=int(core.sum()),write=int(write.sum()),model=int(model.sum()),protected=int(protect.sum()),projected_area=v['projected_areas'][f]))
                if core.any():asset_candidates.append(dict(frame=f,camera=c,core_pixels=int(core.sum()),projected_area=v['projected_areas'][f]))
                if f in [0,14,29]:
                    row=[0,14,29].index(f);rgb=np.array(Image.open(dest/'rgb'/f'{f:05}.png'));views=[rgb.copy(),rgb.copy(),rgb.copy()];views[1][write]=(.5*views[1][write]+.5*np.array([255,190,0])).astype('uint8');views[2][model]=(.5*views[2][model]+.5*np.array([60,160,255])).astype('uint8');views[2][protect]=[10,230,180]
                    for col,im in enumerate(views):sheet.paste(Image.fromarray(im).resize((512,288)),(512*col,322*row+30));draw.text((512*col+5,322*row+6),f"{s['name']} target{s['actor']} CAM{c} f{f} | {['original','write','model/keep'][col]}",fill='white')
            sheet.save(dest/'mask_review.jpg',quality=94);dump(dest/'mask_stats.json',stats)
            row=dict(scene=s['name'],camera=c,seconds=time.time()-started,empty_core_frames=[r['frame'] for r in stats if r['core']==0]);state['completed'].append(row);dump(ROOT/'mask_state.json',state);progress('mask',**row)
            del track;torch.cuda.empty_cache()
        asset=base/'asset';asset.mkdir()
        if asset_candidates:
            # 同一可复用规则：30帧六相机中可见实例core面积最大的实际参考，无结果后选图。
            src=max(asset_candidates,key=lambda r:r['core_pixels']);c=src['camera'];f=src['frame'];raw=Image.open(Path(s['root'])/'images'/f'{f:03}_{c}.jpg').convert('RGB');m=np.array(Image.open(base/f'cam{c}/core'/f'{f:05}.png').resize(raw.size,Image.Resampling.NEAREST))>0;yy,xx=np.where(m);box=[max(0,int(xx.min())-12),max(0,int(yy.min())-12),min(raw.width,int(xx.max())+13),min(raw.height,int(yy.max())+13)]
            Image.fromarray(np.dstack([np.array(raw),m.astype('uint8')*255])).crop(box).save(asset/'source_rgba.png');raw.crop(box).save(asset/'source_context.png')
            dump(asset/'registration.json',dict(task_id=reg['task_id'],run_id='r1',scene=s['name'],actor=s['actor'],source=src,crop=box,fixed=reg['fixed'],source_rule='max visible core pixels in30x6, before asset model',human_verdict=None,failure_ledger_refs=['V77-F02']))
        else:dump(asset/'registration.json',dict(scene=s['name'],actor=s['actor'],state='no_instance_reference',human_verdict=None))
state['state']='complete';dump(ROOT/'mask_state.json',state);progress('mask',state='complete',streams=len(state['completed']))
