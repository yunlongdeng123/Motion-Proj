"""V76-F02唯一预注册控制：可见检测与三维track关联；固定12视图、CPU。"""
import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

ROOT=Path('/root/autodl-tmp/data/v76_vadgs')
OUT=ROOT/'visible_detection_gate'
WEIGHTS=Path('/root/autodl-tmp/models/worldsim_v75/evaluators/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth')
POSITIVE=[('scene_0230','020_0',37),('scene_0230','020_1',34),('scene_0230','020_3',34),
          ('scene_0230','020_5',29),('scene_0230','020_5',8),('scene_0255','020_5',3)]
NEGATIVE=[('scene_0230','020_3',6),('scene_0230','020_3',21),('scene_0255','020_2',14),('scene_0255','020_2',23)]


def coco_class(name):
    if name.startswith('human.pedestrian'): return 1
    if name.startswith('vehicle.bus'): return 6
    return {'vehicle.car':3,'vehicle.truck':8,'vehicle.motorcycle':4,'vehicle.bicycle':2}.get(name)


def iou(a,b):
    a,b=np.array(a),np.array(b)
    intersection=np.prod(np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0))
    union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return float(intersection/max(union,1e-8))


def main():
    OUT.mkdir(exist_ok=True)
    protocol={'task_id':'VADGS-VISIBLE-DETECTION-GATE-20260926',
        'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'model':'torchvision 0.16.2 FasterRCNN ResNet50 FPN v2 COCO_V1',
        'weights':str(WEIGHTS),'device':'cpu','cpu_threads':2,'seed':0,
        'score_threshold':.25,'iou_threshold':.3,
        'threshold_provenance':'existing V75 evaluate_localization.py; unchanged before inference',
        'association':'same COCO class, maximum IoU one-to-one assignment, unmatched allowed',
        'positive_controls':POSITIVE,'negative_controls':NEGATIVE,
        'gate':'all six visible controls associated; all four foreground-confusion controls unmatched',
        'limits':'development controls chosen from previous gate images, not an independent test; remaining prompts diagnostic only',
        'failure_ledger_refs':['V76-F02'],'failure_ledger_delta':'none'}
    with (OUT/'protocol.json').open('x') as handle:
        json.dump(protocol,handle,indent=2)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None).eval()
    model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    rows=[]
    selected={}
    start=time.monotonic()
    with torch.inference_mode():
        for scene_name in ('scene_0230','scene_0255'):
            scene=ROOT/scene_name
            tracks=json.loads((scene/'instances/instances_info.json').read_text())
            views=json.loads((scene/'sam_prior_evidence/generation_report.json').read_text())['views']
            for view in views:
                name=view['name']
                rgb=Image.open(scene/'images'/f'{name}.jpg').convert('RGB')
                tensor=torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).float()/255
                pred=model([tensor])[0]
                candidates=[{'box':box.tolist(),'score':float(score),'label':int(label)}
                    for box,score,label in zip(pred['boxes'],pred['scores'],pred['labels']) if float(score)>=.25]
                objects=view['projected_objects']
                compatibility=np.zeros((len(objects),len(candidates)),dtype=float)
                matches={}
                for i,obj in enumerate(objects):
                    category=coco_class(tracks[str(obj['track_id'])]['class_name'])
                    for j,det in enumerate(candidates):
                        if det['label']==category:
                            overlap=iou(obj['box_xyxy'],det['box'])
                            if overlap>=.3:
                                compatibility[i,j]=overlap
                if compatibility.size:
                    ii,jj=linear_sum_assignment(-compatibility)
                    for i,j in zip(ii,jj):
                        if compatibility[i,j]>0:
                            matches[i]=dict(detection=candidates[j],iou=float(compatibility[i,j]),candidate_index=int(j))
                associations=[]
                picture=rgb.copy()
                draw=ImageDraw.Draw(picture)
                for i,obj in enumerate(objects):
                    key=(scene_name,name,obj['track_id'])
                    selected[key]=i in matches
                    association=dict(track_id=obj['track_id'],class_name=tracks[str(obj['track_id'])]['class_name'],
                        projected_box=obj['box_xyxy'],match=matches.get(i))
                    associations.append(association)
                    draw.rectangle(obj['box_xyxy'],outline='lime' if i in matches else 'red',width=3)
                    draw.text((obj['box_xyxy'][0],obj['box_xyxy'][1]-13),f"ID{obj['track_id']} {'MATCH' if i in matches else 'NONE'}",fill='yellow')
                    if i in matches:
                        draw.rectangle(matches[i]['detection']['box'],outline='cyan',width=2)
                picture.resize((960,540)).save(OUT/f'{scene_name}_{name}.jpg')
                row={'scene':scene_name,'view':name,'detections':candidates,'associations':associations}
                rows.append(row)
                (OUT/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(f'{scene_name} {name}: {len(matches)}/{len(objects)} associated',flush=True)
    positives=[{'case':key,'matched':selected.get(key,False)} for key in POSITIVE]
    negatives=[{'case':key,'matched':selected.get(key,False)} for key in NEGATIVE]
    passed=all(row['matched'] for row in positives) and not any(row['matched'] for row in negatives)
    result={'task_id':protocol['task_id'],'passed_detection_gate':passed,'positive_controls':positives,
        'negative_controls':negatives,'views':rows,'seconds':time.monotonic()-start,
        'sam_rerun_performed':False,'training_input_approved':False,
        'failure_ledger_refs':['V76-F02'],'failure_ledger_delta':'none; fixed visibility-control result added'}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='views'}),flush=True)


if __name__=='__main__': main()
