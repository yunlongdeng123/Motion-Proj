#!/usr/bin/env python3
"""Preserved DEVA defects -> real cached SAM candidates; root review remains pending."""
import argparse, fcntl, json, os, subprocess, sys, threading, time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT=Path('/root/autodl-tmp/drivingeditor_repro')
F=ROOT/'campaign/followup_scene_prep_20261013'
SCENE=F/'inputs/scene0029_069_117_red_cpu_v1'
OUT=F/'metadata/red_sam_candidates_20261013'
def nv():
    kw={'capture_output':True,'text':True,'timeout':8}
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],**kw)
    if p.returncode: raise RuntimeError(p.stderr)
    dev=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],**kw)
    if dev.returncode: raise RuntimeError(dev.stderr)
    procs=[]
    for line in p.stdout.splitlines():
        if line.strip():
            pid,mb=line.split(','); procs.append({'pid':int(pid),'bytes':int(mb)*1048576})
    return procs,[int(x)*1048576 for x in dev.stdout.splitlines() if x.strip()]
def main():
    p=argparse.ArgumentParser();p.add_argument('--frames',nargs='+',required=True);a=p.parse_args()
    assert len(a.frames)==len(set(a.frames)) and all(x.isdigit() and len(x)==7 for x in a.frames)
    with (ROOT/'campaign/gpu_slot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        procs,devices=nv();assert not procs, 'Unrelated GPU compute exists; preserve and stop'
        assert not OUT.exists(),'No-clobber: candidate directory already exists'
        OUT.mkdir(); samples=[];stop=threading.Event()
        def sampler():
            while not stop.is_set():
                try:
                    procs,dev=nv();samples.append({'time':time.time(),'processes':procs,'devices':dev})
                except Exception as e:samples.append({'time':time.time(),'error':str(e)})
                stop.wait(1)
        thread=threading.Thread(target=sampler,daemon=True);thread.start()
        import numpy as np
        from PIL import Image,ImageDraw
        import torch
        from segment_anything import sam_model_registry,SamPredictor
        torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
        checkpoint=ROOT/'weights/sam_vit_h_4b8939.pth'
        sam=sam_model_registry['vit_h'](checkpoint=str(checkpoint)).to('cuda').eval()
        predictor=SamPredictor(sam); rows=[];start=time.time()
        try:
            with torch.inference_mode():
                for frame in a.frames:
                    rgb=np.array(Image.open(SCENE/'image/image_0'/f'{frame}.png').convert('RGB'))
                    ids=np.array(Image.open(SCENE/'mask/mask_0/Annotations'/f'{frame}.png'))
                    body=ids==130;y,x=np.where(body);assert len(x)>0
                    # Box derives only from real DEVA support with fixed80px padding, no GT boxes.
                    box=np.array([max(0,int(x.min())-80),max(0,int(y.min())-80),min(1919,int(x.max())+80),min(1279,int(y.max())+80)],dtype=np.float32)
                    predictor.set_image(rgb)
                    masks,scores,logits=predictor.predict(box=box,multimask_output=True)
                    candidates=[]
                    for i,mask in enumerate(masks):
                        path=OUT/f'{frame}_candidate{i}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
                        candidates.append({'candidate_index':i,'path':str(path),'predicted_iou':float(scores[i]),'area':int(mask.sum()),'deva_overlap_pixels':int((mask&body).sum())})
                    # Native pixel copies, labels outside tiles; no mask editing or exposure scaling.
                    cy0=max(0,int(y.min())-100);cy1=min(1280,int(y.max())+100)
                    cx0=max(0,int(x.min())-130);cx1=1920
                    tiles=[rgb,np.repeat((body*255).astype(np.uint8)[...,None],3,axis=2)]+[rgb*mask[...,None] for mask in masks]
                    tilew=cx1-cx0;tileh=cy1-cy0
                    contact=Image.new('RGB',(tilew*5,tileh+28),(20,20,20));draw=ImageDraw.Draw(contact)
                    for i,tile in enumerate(tiles):
                        contact.paste(Image.fromarray(tile[cy0:cy1,cx0:cx1]),(i*tilew,28));draw.text((i*tilew+2,5),['RGB','DEVA130','SAM0','SAM1','SAM2'][i],fill='white')
                    contact.save(OUT/f'{frame}_candidate_pixelcopy.png')
                    rows.append({'frame_id':frame,'prompt':{'kind':'real_DEVA130_bbox_fixed80px_padding','box_xyxy':box.tolist(),'GT_boxes_used':False},'model_output_candidates':candidates})
                    print(json.dumps(rows[-1]),flush=True)
            result={'status':'REAL_SAM_CANDIDATES_EXPORTED_REVIEW_PENDING','scene':str(SCENE),'source_script':str(Path(__file__).resolve()),'interpreter':sys.executable,'checkpoint':str(checkpoint),'returncode':0,'frames':rows,'model':'cached_SAM_vit_h','amp':False,'source_DEVA_modified':False,'postprocessing':'none; exact boolean model output','GT_boxes_used':False,'root_review_complete':False,'human_verdict':None,'training_ready':False,'wall_seconds':time.time()-start,'torch_allocated_peak_bytes':torch.cuda.max_memory_allocated(),'torch_reserved_peak_bytes':torch.cuda.max_memory_reserved()}
        finally:
            stop.set();thread.join(timeout=10)
            (OUT/'memory_samples.json').write_text(json.dumps(samples,indent=2))
        result['nvidia_sample_count']=len(samples);result['nvidia_sampling_errors']=sum('error'in x for x in samples)
        result['nvidia_process_peak_bytes']=max((p['bytes'] for s in samples for p in s.get('processes',[]) if p['pid']==os.getpid()),default=None)
        result['nvidia_device_peak_bytes']=max((v for s in samples for v in s.get('devices',[])),default=None)
        result['nvidia_peaks_are_sampled_lower_bounds']=True
        (OUT/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'frames':len(rows),'receipt':str(OUT/'status.json')}),flush=True)
if __name__=='__main__':main()
