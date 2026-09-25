"""固定12视图重读未裁切SAM响应，检查框提示是否选中了遮挡物。"""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from generate_sam_prior import SAM_ROOT, DEFAULT_CKPT

sys.path.insert(0,str(SAM_ROOT/'segment_anything'))
from segment_anything import sam_model_registry, SamPredictor
torch.set_num_threads(2)
predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=str(DEFAULT_CKPT)).cuda().eval())
root = Path('/root/autodl-tmp/data/v76_vadgs')
output = root/'sam_occlusion_audit'
output.mkdir(exist_ok=True)
rows = []
for name in ('scene_0230','scene_0255'):
    scene = root/name
    tracks = json.loads((scene/'instances/instances_info.json').read_text())
    report = json.loads((scene/'sam_prior_evidence/generation_report.json').read_text())
    for view in report['views']:
        if not view['projected_objects']:
            continue
        image = cv2.imread(str(scene/'images'/f'{view["name"]}.jpg'))
        predictor.set_image(image[...,::-1].copy())
        for obj in view['projected_objects']:
            box = np.array(obj['box_xyxy'])
            masks,scores,_ = predictor.predict(box=box,multimask_output=False)
            mask = masks[0]
            x0,y0,x1,y1 = np.r_[np.floor(box[:2]),np.ceil(box[2:])].astype(int)
            inside = np.zeros(mask.shape,dtype=bool)
            inside[y0:y1+1,x0:x1+1] = True
            support = int(mask.sum())
            row = dict(scene=name,view=view['name'],track_id=obj['track_id'],
                class_name=tracks[str(obj['track_id'])]['class_name'],score=float(scores[0]),
                raw_pixels=support,inside_pixels=int((mask&inside).sum()),
                outside_fraction=float((mask&~inside).sum()/max(1,support)),
                box_area=int(inside.sum()))
            row['mask_larger_than_box'] = support > row['box_area']
            rows.append(row)
            cv2.imwrite(str(output/f'{name}_{view["name"]}_id{obj["track_id"]}_raw.png'),mask.astype(np.uint8)*255)
            if row['mask_larger_than_box']:
                overlay = image.copy()
                overlay[mask] = (.5*overlay[mask]+.5*np.array([0,0,255])).astype(np.uint8)
                cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,255,255),3)
                cv2.imwrite(str(output/f'{name}_{view["name"]}_id{obj["track_id"]}_overlay.jpg'),overlay)
predictor.reset_image()
payload = {'task_id':'VADGS-MATCHED-SAM-OCCLUSION-20260926','views':12,'objects':rows,
           'note':'raw SAM masks before AABB clipping; area outside the target projection is an identity consistency diagnostic'}
(output/'audit.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(rows))
