"""Frozen, offline Cityscapes inference. No boxes or reconstruction as labels."""
import argparse,json,os,sys,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import SegformerForSemanticSegmentation,SegformerImageProcessor
OLD=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
ROOT=OLD.parent/'v43_ownership_20261028'
sys.path.insert(0,str(OLD/'code'))
from gpu_slot import exclusive
CONTEXTS=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']
NAMES=['unknown','road','sidewalk','building','wall','fence','pole','traffic_light','traffic_sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
COLORS=np.array([[80,89,104],[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],[0,80,100],[0,0,230],[119,11,32]],np.uint8)
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
    a=argparse.ArgumentParser();a.add_argument('--model',default='/root/autodl-tmp/models/worldsim_v6/r20_semantic_segformer_cityscapes');a.add_argument('--attempt',default='segformer_b0_v1');args=a.parse_args()
    dest=ROOT/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();status=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase='waiting_GPU',frames=0,model=args.model)
    def update(**kw):status.update(kw);write(dest/'status.json',status)
    update()
    with exclusive():
        torch.set_num_threads(2);update(phase='loading_model')
        processor=SegformerImageProcessor.from_pretrained(args.model,local_files_only=True)
        model=SegformerForSemanticSegmentation.from_pretrained(args.model,local_files_only=True).cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        config={str(k):v for k,v in model.config.id2label.items()};assert len(config)==19
        write(dest/'semantic_map.json',dict(version='cityscapes19-plus-unknown-v1',ids=dict(enumerate(NAMES)),source_id_offset=1,probability_threshold=.65,model=args.model,source_labels=config,not_ground_truth=True))
        for ctx in CONTEXTS:
            source=OLD/'motion_scene_inputs_v1'/ctx;sensor=json.loads((source/'sensor_preparation.json').read_text());out=dest/ctx;out.mkdir();records=[]
            update(phase='inference',context=ctx)
            for i,f in enumerate(sensor['frames']):
                paths=[source/'image/image_0'/(f['frame_id']+ext) for ext in ['.jpg','.png','.jpeg']]
                image=Image.open(next(p for p in paths if p.exists())).convert('RGB')
                with torch.no_grad():
                    inputs=processor(images=image,return_tensors='pt');inputs={k:v.cuda() for k,v in inputs.items()}
                    logits=model(**inputs).logits.float();logits=torch.nn.functional.interpolate(logits,(320,480),mode='bilinear',align_corners=False)
                    prob=logits.softmax(1);conf,pred=prob.max(1);pred=pred[0].cpu().numpy().astype('uint8')+1;conf=conf[0].cpu().numpy()
                labels=pred.copy();labels[conf<.65]=0
                np.savez_compressed(out/('%03d.npz'%f['raw']),labels=labels,prediction=pred,confidence=conf.astype(np.float16))
                Image.fromarray(labels).save(out/('%03d_labels.png'%f['raw']))
                if i in [0,12,24,36,48]:
                    rgb=np.asarray(image.resize((480,320)));vis=(rgb*.6+COLORS[labels]*.4).astype('uint8');Image.fromarray(vis).save(out/('%03d_overlay.png'%f['raw']))
                records.append(dict(raw=f['raw'],frame_id=f['frame_id'],unknown_fraction=float((labels==0).mean()),class_counts=np.bincount(labels.ravel(),minlength=20).tolist()))
                update(frames=status['frames']+1,last_raw=f['raw']);print('SEGMENT',ctx,f['raw'],round(time.monotonic()-started,2),flush=True)
            write(out/'summary.json',dict(frames=records,model=args.model,model_trained=False,prediction_not_GT=True))
        update(phase='complete',wall_seconds=time.monotonic()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),new_training_updates=0)
if __name__=='__main__':main()
