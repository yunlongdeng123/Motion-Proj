"""统一SAM2实例规则；保留raw/core和逐帧输入，先交图像检查。"""
from pathlib import Path
import sys,os,time,json
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump,hull_mask,camera,largest
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1')
def main():
    assert not (ROOT/'mask_state.json').exists();state=dict(state='loading',pid=os.getpid(),completed=[]);dump(ROOT/'mask_state.json',state);reg=read(ROOT/'registration.json');torch.set_num_threads(4);cv2.setNumThreads(4);torch.manual_seed(42)
    sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2');from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for s in reg['new_scenes']:
            frames=read(ROOT/s['name']/'frames.json')
            for v in s['streams']:
                base=ROOT/s['name']/f"cam{v['camera']}";raw={};beg=time.time();track=predictor.init_state(video_path=str(base/'rgb'),offload_video_to_cpu=True,offload_state_to_cpu=True);j=v['prompt_frame'];predictor.add_new_points_or_box(track,frame_idx=j,obj_id=1,box=np.array(v['boxes'][j],np.float32))
                for reverse in [False,True]:
                    for f,_,logit in predictor.propagate_in_video(track,start_frame_idx=j,reverse=reverse):raw[f]=(logit[0,0]>0).cpu().numpy()
                assert len(raw)==30
                for folder in ['sam','core','write_mask','model_mask','protect','alpha']:(base/folder).mkdir()
                rows=[];sheet=Image.new('RGB',(1536,322*3),(20,30,40));d=ImageDraw.Draw(sheet)
                for f,fr in enumerate(frames):
                    c,k=camera(fr,v['camera']);b=next(b for b in fr['all_boxes'] if b['actor_id']==s['actor']);gate=hull_mask(b,c,k,pad=3);core=largest(raw[f]&gate);write=cv2.dilate(core.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))>0;write &= gate;model=np.zeros((576,1024),bool)
                    if write.any():
                        yy,xx=np.where(write);model[max(0,yy.min()-8):min(576,yy.max()+25),max(0,xx.min()-8):min(1024,xx.max()+9)]=True
                    protect=np.zeros_like(model)
                    for other in fr['all_boxes']:
                        if other['actor_id']!=s['actor']:protect |= hull_mask(other,c,k)
                    protect &= model&~write;dist=cv2.distanceTransform(model.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);t=np.clip(dist/8,0,1);alpha=t*t*(3-2*t);alpha[write]=1;alpha[protect]=0
                    rgb=np.array(Image.open(Path(s['root'])/'images'/f"{f:03}_{v['camera']}.jpg").convert('RGB').resize((1024,576),Image.Resampling.BILINEAR));Image.fromarray(rgb).save(base/'rgb'/f'{f:05}.png')
                    for folder,arr in [('sam',raw[f]*255),('core',core*255),('write_mask',write*255),('model_mask',model*255),('protect',protect*255),('alpha',np.rint(alpha*255))]:Image.fromarray(arr.astype('uint8')).save(base/folder/f'{f:05}.png')
                    rows.append(dict(frame=f,raw_sam=int(raw[f].sum()),core=int(core.sum()),write=int(write.sum()),model=int(model.sum()),protected=int(protect.sum()),projected_area=v['projected_areas'][f]))
                    if f in [0,14,29]:
                        row=[0,14,29].index(f);views=[rgb.copy(),rgb.copy(),rgb.copy()];views[1][write]=(.5*views[1][write]+.5*np.array([255,190,0])).astype('uint8');views[2][model]=(.5*views[2][model]+.5*np.array([60,160,255])).astype('uint8');views[2][protect]=[10,230,180]
                        for col,im in enumerate(views):sheet.paste(Image.fromarray(im).resize((512,288)),(512*col,322*row+30));d.text((512*col+5,322*row+6),f"{s['name']} actor{s['actor']} CAM{v['camera']} f{f} | {['original','target instance write','model / keep'][col]}",fill='white')
                sheet.save(base/'mask_review.jpg',quality=94);dump(base/'mask_stats.json',rows);state['completed'].append(dict(scene=s['name'],camera=v['camera'],seconds=time.time()-beg,empty_core_frames=[r['frame'] for r in rows if r['core']==0],human_verdict=None));state['state']='running';dump(ROOT/'mask_state.json',state);print('MASK_DONE',s['name'],v['camera'],time.time()-beg,flush=True);del track;torch.cuda.empty_cache()
    state['state']='complete_pending_visual_review';dump(ROOT/'mask_state.json',state)
if __name__=='__main__':main()
