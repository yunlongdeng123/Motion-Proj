"""Additional first-scene evidence; frozen model, preserve previous49 labels."""
import json,os,time
from pathlib import Path
import numpy as np
from PIL import Image
from segmentation import ROOT,write,SegformerForSemanticSegmentation,SegformerImageProcessor,torch
from gpu_slot import exclusive
from v43_resource_control import require_start
CTX='14705303724557273004_3105_000_3125_000'

def main():
    require_start(CTX);R=ROOT/'target_unknown_v2';D=R/'fullclip_semantics';D.mkdir(exist_ok=False)
    while not (R/'fullclip_inputs/manifest.json').exists():
        require_start(CTX);time.sleep(2)
    frames=json.loads((R/'fullclip_inputs/manifest.json').read_text())['frames'];started=time.monotonic()
    with exclusive():
        torch.set_num_threads(1);path='/root/autodl-tmp/models/worldsim_v6/r20_semantic_segformer_cityscapes';processor=SegformerImageProcessor.from_pretrained(path,local_files_only=True);model=SegformerForSemanticSegmentation.from_pretrained(path,local_files_only=True).cuda().eval();records=[]
        with torch.no_grad():
            for f in frames:
                require_start(CTX);raw=f['raw'];old=ROOT/'segformer_b0_attempt3'/CTX/('%03d.npz'%raw)
                if old.exists():(D/old.name).write_bytes(old.read_bytes());reused=True
                else:
                    image=Image.open(R/'fullclip_inputs/rgb'/('%03d.png'%raw));inputs={k:v.cuda() for k,v in processor(images=image,return_tensors='pt').items()};logits=torch.nn.functional.interpolate(model(**inputs).logits.float(),(320,480),mode='bilinear',align_corners=False);conf,pred=logits.softmax(1).max(1);pred=pred[0].cpu().numpy().astype('uint8')+1;conf=conf[0].cpu().numpy();labels=pred.copy();labels[conf<.65]=0;np.savez_compressed(D/('%03d.npz'%raw),labels=labels,prediction=pred,confidence=conf.astype('float16'));reused=False
                records.append(dict(raw=raw,reused_previous49=reused));write(D/'status.json',dict(phase='segmenting',frames=len(records),pid=os.getpid()));print('FULL_SEM',raw,flush=True)
    write(D/'summary.json',dict(context=CTX,frames=records,model_frozen=True,old49_preserved=True,additional_frames_input_size=[480,320],additional_preprocessing_different_from_original1920=True,not_GT=True,wall_seconds=time.monotonic()-started));write(D/'status.json',dict(phase='complete',frames=len(frames)))

if __name__=='__main__':main()
