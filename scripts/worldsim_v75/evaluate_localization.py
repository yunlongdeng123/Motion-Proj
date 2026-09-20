"""独立COCO检测器测量投影变化；只报告二维框，不能冒充米制状态真值。"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from common import RUN, ASSETS
from select_target import P1

FRAMES=[0,5,13,21,29,36,37,45,53]
WEIGHTS=ASSETS/'evaluators/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth'
def iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/max(union,1e-8)

def match(pred,bounds):
    rows=[]
    for b,s,l in zip(pred['boxes'],pred['scores'],pred['labels']):
        if int(l)!=3 or float(s)<0.25:
            continue
        box=b.tolist()
        rows.append({'box':box,'score':float(s),'iou_reference':iou(box,bounds),
                     'center':[(box[0]+box[2])/2,(box[1]+box[3])/2]})
    if not rows:
        return None
    best=max(rows,key=lambda r:r['iou_reference'])
    return best if best['iou_reference']>=0.3 else None

def model():
    torch.set_num_threads(4)
    m=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None).eval()
    m.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    return m

def predict(m,rgb):
    t=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).float()/255
    with torch.inference_mode():
        return m([t])[0]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','evaluate'])
    p.add_argument('--case',default='clean')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.phase=='prepare':
        config_path=P1/'evaluator_protocol.json'
        if config_path.exists():
            raise FileExistsError(config_path)
        config={'frozen_utc':datetime.now(timezone.utc).isoformat(),
                'model':'torchvision FasterRCNN ResNet50 FPN v2 COCO_V1','source_url':FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url,
                'device':'cpu','class_id':3,'score_threshold':0.25,'matching_iou_threshold':0.3,
                'frames':FRAMES,'clean_required_matches':6,'calibration_shifts_px':[-4,4],
                'calibration_max_abs_residual_px':2.0,'biased_video_inspected':False,
                'note':'附加评价器在偏置输出查看前冻结。参考投影只用于配对关联；未来像素无GT，不输出米制误差。'}
        config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
        WEIGHTS.parent.mkdir(exist_ok=True)
        if not WEIGHTS.exists():
            torch.hub.download_url_to_file(config['source_url'],str(WEIGHTS),progress=False)
        m=model()
        original=np.asarray(Image.open(RUN/'initial_rgb.png'))
        reference=json.loads((P1/'target_projections.json').read_text())['clean'][0]['bounds']
        base=match(predict(m,original),reference)
        rows=[]
        for shift in [-4,4]:
            changed=np.zeros_like(original)
            if shift>0:
                changed[:,shift:]=original[:,:-shift]
            else:
                changed[:,:shift]=original[:,-shift:]
            bounds=np.asarray(reference)+np.array([shift,0,shift,0])
            matched=match(predict(m,changed),bounds.tolist())
            residual=None if base is None or matched is None else float(matched['center'][0]-base['center'][0]-shift)
            rows.append({'shift_x_px':shift,'match':matched,'center_translation_residual_px':residual})
        passed=base is not None and all(r['center_translation_residual_px'] is not None and abs(r['center_translation_residual_px'])<=2.0 for r in rows)
        result={'status':'passed' if passed else 'failed','real_initial_match':base,'calibration':rows,
                'meaning':'检测器对真实初帧的已知像素平移检验；不是未来真值评价'}
        (P1/'evaluator_calibration.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    stem=f'{a.case}-seed{a.seed}'
    run=json.loads((P1/'rollouts'/f'{stem}.json').read_text())
    assert run['status']=='complete'
    video=np.load(P1/'rollouts'/f'{stem}.npy',mmap_mode='r')
    projection=json.loads((P1/'target_projections.json').read_text())['clean']
    m=model()
    rows=[]
    for f in FRAMES:
        result=match(predict(m,video[f]),projection[f]['bounds'])
        rows.append({'frame':f,'match':result})
        print(json.dumps({'case':stem,'frame':f,'match':result}),flush=True)
    (P1/'rollouts'/f'{stem}.detections.json').write_text(json.dumps({'case':a.case,'seed':a.seed,'frames':rows,
        'matched':sum(r['match'] is not None for r in rows),'required_clean_matches':6},indent=2)+'\n')

if __name__=='__main__':
    main()
