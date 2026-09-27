"""固定十时刻语义检查：只提示疑似再生，不把真实隐藏邻车自动判幻觉。"""
from pathlib import Path
import sys,time,numpy as np,torch,cv2
from PIL import Image,ImageDraw
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read,dump
from repair_guard import classify
sys.path.insert(0,'/root/autodl-tmp/third_party/worldsim_v32/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
R=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1')

def main():
    assert read(R/'drive_state.json')['state']=='complete_pending_visual_review';dest=R/'guard';dest.mkdir(exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(4)
    m='/root/autodl-tmp/models/worldsim_v77/grounding-dino-tiny';processor=AutoProcessor.from_pretrained(m,local_files_only=True);model=AutoModelForZeroShotObjectDetection.from_pretrained(m,local_files_only=True).eval().cuda();sam=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
    def detect(im):
        inp=processor(images=Image.fromarray(im),text='car. truck. bus. van.',return_tensors='pt').to('cuda')
        with torch.inference_mode():pred=model(**inp)
        r=processor.post_process_grounded_object_detection(pred,inp.input_ids,box_threshold=.25,text_threshold=.25,target_sizes=[im.shape[:2]])[0];boxes=r['boxes'].cpu().numpy()
        if len(boxes)==0:return []
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):sam.set_image(im);m,_,_=sam.predict(box=boxes,multimask_output=False)
        if m.ndim==4:m=m[:,0]
        return [dict(box=b.tolist(),score=float(sc),mask=mm>0) for b,sc,mm in zip(boxes,r['scores'],m)]
    rows=[];start=time.time()
    for scene,cam in read(R/'mask_admission.json')['admitted_primary_streams']:
        base=R/scene/f'cam{cam}';out=dest/scene;out.mkdir();records=[]
        for f in np.linspace(0,29,10).round().astype(int):
            orig=np.array(Image.open(base/'rgb'/f'{f:05}.png').convert('RGB'));generated=np.array(Image.open(base/'drive/composite'/f'{f:05}.png').convert('RGB'));core=np.array(Image.open(base/'core'/f'{f:05}.png'))>0;edit=np.array(Image.open(base/'model_mask'/f'{f:05}.png'))>0;src=detect(orig);got=detect(generated)
            target=[v for v in src if (v['mask']&core).sum()/max(1,core.sum())>.3];neighbors=[v['mask'] for v in src if (v['mask']&core).sum()/max(1,core.sum())<=.05]
            details=[dict(box=v['box'],score=v['score'],**classify(v['mask'],core,edit,neighbors)) for v in got];row=dict(frame=int(f),source_target_positive=bool(target),suspect=any(v['suspect_new_vehicle'] for v in details),detections=details);records.append(row)
            np.savez_compressed(out/f'{f:05}.npz',**{f'source_{j}':v['mask'] for j,v in enumerate(src)},**{f'output_{j}':v['mask'] for j,v in enumerate(got)})
            im=Image.fromarray(generated);d=ImageDraw.Draw(im)
            for v in details:
                if v['suspect_new_vehicle']:d.rectangle(v['box'],outline='red',width=2)
            im.save(out/f'{f:05}.jpg',quality=93)
        summary=dict(scene=scene,frames=10,source_positive=sum(r['source_target_positive'] for r in records),flagged_frames=[r['frame'] for r in records if r['suspect']],records=records,human_verdict=None);dump(out/'summary.json',summary);rows.append(summary);print('GUARD_DONE',scene,summary['source_positive'],summary['flagged_frames'],flush=True)
    dump(dest/'summary.json',dict(scenes=rows,seconds=time.time()-start,model='existing GroundingDINO-tiny + SAM2.1-large',scope='10/30 frames, advisory flags require identity/visibility review; zero detections not background pass. Never equate reappearing true hidden neighbor with hallucination.',human_verdict=None))
if __name__=='__main__':main()
