"""固定0/40/60帧扩展V76-F02控制；CPU、隔离输出、不调检测或关联参数。"""
import datetime
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.optimize import linear_sum_assignment
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from generate_sam_prior import dynamic_objects, project_box, SAM_ROOT, DEFAULT_CKPT
from test_visible_detection_gate import coco_class, iou, WEIGHTS

ROOT = Path('/root/autodl-tmp/data/v76_vadgs')
OUT = ROOT/'visible_sam_crossframe_gate'
SCENES = ('scene_0230', 'scene_0255')
FRAMES = (0, 40, 60)


def main():
    OUT.mkdir(exist_ok=False)
    protocol = {
        'task_id': 'VADGS-VISIBLE-SAM-CROSSFRAME-20260926',
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'frames': FRAMES, 'cameras': list(range(6)), 'scenes': SCENES,
        'selection': 'fixed start, later interior and last frame, all cameras; frame20 already used for development',
        'score_threshold': .25, 'iou_threshold': .3,
        'association': 'same COCO class maximum IoU one-to-one; unmatched allowed',
        'detector': str(WEIGHTS), 'sam': str(DEFAULT_CKPT),
        'device': 'cpu', 'cpu_threads': 2, 'seed': 0,
        'review': 'inspect every projected object crop for identity mistakes and visible unmatched targets; include all views in contact sheets',
        'stop_rule': 'any clear foreground identity error or visible unmatched dynamic target prevents full-input qualification; no threshold sweep or replacement detector',
        'scope': 'adapted data development control, not an independent method test or human verdict',
        'training_inputs_modified': False, 'failure_ledger_refs': ['V76-F02'],
    }
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None).eval()
    detector.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True), strict=True)
    sys.path.insert(0, str(SAM_ROOT/'segment_anything'))
    from segment_anything import sam_model_registry, SamPredictor
    predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=str(DEFAULT_CKPT)).eval())
    rows, crops = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for scene_name in SCENES:
            scene = ROOT/scene_name
            actors = dynamic_objects(scene)
            tracks = json.loads((scene/'instances/instances_info.json').read_text())
            destination = OUT/scene_name
            destination.mkdir()
            for frame in FRAMES:
                sheet = Image.new('RGB', (1200, 6*360), 'white')
                for camera in range(6):
                    name = f'{frame:03d}_{camera}'
                    rgb = np.array(Image.open(scene/'images'/f'{name}.jpg').convert('RGB'))
                    h, w = rgb.shape[:2]
                    intrinsics = np.loadtxt(scene/'intrinsics'/f'{camera}.txt')
                    w2c = np.linalg.inv(np.loadtxt(scene/'extrinsics'/f'{name}.txt'))
                    objects = []
                    for key, ann in actors.items():
                        if frame not in ann:
                            continue
                        projection = project_box(*ann[frame], w2c, intrinsics, w, h)
                        if projection is not None:
                            box, depth = projection
                            objects.append({'track_id': key, 'class_name': tracks[str(key)]['class_name'],
                                            'projected_box': box.tolist(), 'center_depth': depth})
                    pred = detector([torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255])[0]
                    detections = [{'box': b.tolist(), 'score': float(s), 'label': int(c)}
                                  for b,s,c in zip(pred['boxes'], pred['scores'], pred['labels']) if s >= .25]
                    matrix = np.zeros((len(objects),len(detections)))
                    for i, obj in enumerate(objects):
                        for j, det in enumerate(detections):
                            overlap = iou(obj['projected_box'], det['box'])
                            if det['label'] == coco_class(obj['class_name']) and overlap >= .3:
                                matrix[i,j] = overlap
                    matches = {}
                    if matrix.size:
                        ii, jj = linear_sum_assignment(-matrix)
                        matches = {int(i): int(j) for i,j in zip(ii,jj) if matrix[i,j] > 0}
                    ids = np.full((h,w,3), 255, dtype=np.uint8)
                    if matches:
                        predictor.set_image(rgb)
                    for i in sorted(matches, key=lambda i: detections[matches[i]]['score'], reverse=True):
                        j = matches[i]
                        raw, score, _ = predictor.predict(box=np.array(detections[j]['box']), multimask_output=False)
                        assigned = raw[0] & (ids[...,0] == 255)
                        ids[assigned,0] = objects[i]['track_id']
                        objects[i]['match'] = {'detection': detections[j], 'iou': float(matrix[i,j]),
                                               'raw_pixels': int(raw[0].sum()), 'assigned_pixels': int(assigned.sum()),
                                               'sam_score': float(score[0])}
                        cv2.imwrite(str(destination/f'{name}_id{objects[i]["track_id"]}_raw.png'), raw[0].astype(np.uint8)*255)
                    predictor.reset_image()
                    cv2.imwrite(str(destination/f'{name}_ids.png'), ids)
                    overlay = rgb.copy()
                    for obj in objects:
                        selected = ids[...,0] == obj['track_id']
                        overlay[selected] = (.5*overlay[selected]+.5*np.array([0,160,255])).astype(np.uint8)
                        box = np.array(obj['projected_box']).astype(int)
                        color = (0,230,0) if obj.get('match') else (255,0,0)
                        cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), color, 2)
                        cv2.putText(overlay, str(obj['track_id']), (box[0],max(20,box[1])), cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
                    for obj in objects:
                        x0,y0,x1,y1 = np.array(obj['projected_box']).astype(int)
                        extent = (max(0,x0-40),max(0,y0-40),min(w,x1+40),min(h,y1+40))
                        tile = Image.new('RGB', (1000,280), 'white')
                        label = f'{scene_name} {name} ID{obj["track_id"]} {obj["class_name"]} match={bool(obj.get("match"))}'
                        ImageDraw.Draw(tile).text((5,5),label,fill='black')
                        for column,array in enumerate((rgb,overlay)):
                            crop = ImageOps.contain(Image.fromarray(array).crop(extent), (490,250))
                            tile.paste(crop,(column*500,25))
                        crops.append(tile)
                    for column,array in enumerate((rgb,overlay)):
                        sheet.paste(Image.fromarray(array).resize((600,337)),(column*600,camera*360+23))
                    ImageDraw.Draw(sheet).text((5,camera*360+5),f'{scene_name} {name} : {len(matches)}/{len(objects)} matches',fill='black')
                    rows.append({'scene':scene_name,'view':name,'associations':objects,'detections':detections})
                    (OUT/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
                    print(f'{scene_name} {name}: {len(matches)}/{len(objects)} matches',flush=True)
                sheet.save(destination/f'frame{frame:03d}_all_views.jpg')
    for start in range(0,len(crops),6):
        part = crops[start:start+6]
        page = Image.new('RGB',(1000,280*len(part)),'white')
        for i,tile in enumerate(part):
            page.paste(tile,(0,i*280))
        page.save(OUT/f'objects_{start:03d}.jpg')
    result = {'task_id':protocol['task_id'],'seconds':time.monotonic()-started,'views':rows,
              'view_count':len(rows),'projected_objects':sum(len(v['associations']) for v in rows),
              'matched_objects':sum(bool(a.get('match')) for v in rows for a in v['associations']),
              'visual_review':'pending','training_input_approved':False,
              'failure_ledger_refs':['V76-F02'],'failure_ledger_delta':'none; fixed crossframe extension'}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='views'}),flush=True)


if __name__ == '__main__':
    main()
