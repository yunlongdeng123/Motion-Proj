"""固定12视图的可见检测→SAM控制，CPU上运行，输出隔离sidecar。"""
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from generate_sam_prior import SAM_ROOT, DEFAULT_CKPT

ROOT=Path('/root/autodl-tmp/data/v76_vadgs')
GATE=ROOT/'visible_detection_gate'
OUT=GATE/'sam_r1'
sys.path.insert(0,str(SAM_ROOT/'segment_anything'))
from segment_anything import SamPredictor,sam_model_registry


def main():
    detection=json.loads((GATE/'result.json').read_text())
    assert detection['passed_detection_gate']
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model=sam_model_registry['vit_h'](checkpoint=str(DEFAULT_CKPT)).eval()
    predictor=SamPredictor(model)
    rows=[]
    start=time.monotonic()
    with torch.inference_mode():
        for view in detection['views']:
            scene,name=view['scene'],view['view']
            destination=OUT/scene
            destination.mkdir(exist_ok=True)
            image=cv2.imread(str(ROOT/scene/'images'/f'{name}.jpg'))
            accepted=[a for a in view['associations'] if a['match'] is not None]
            # 可见检测互相冲突时按检测分数先后分配；不制造多对象重叠监督。
            accepted.sort(key=lambda a:a['match']['detection']['score'],reverse=True)
            ids=np.full(image.shape,255,dtype=np.uint8)
            scores=[]
            if accepted:
                predictor.set_image(image[...,::-1].copy())
            for association in accepted:
                bbox=np.array(association['match']['detection']['box'])
                masks,quality,_=predictor.predict(box=bbox,multimask_output=False)
                raw=masks[0]
                mask=raw & (ids[...,0]==255)
                ids[mask,0]=association['track_id']
                scores.append({'track_id':association['track_id'],'raw_pixels':int(raw.sum()),
                    'assigned_pixels':int(mask.sum()),'sam_score':float(quality[0]),
                    'detection':association['match']['detection']})
                cv2.imwrite(str(destination/f'{name}_id{association["track_id"]}_raw.png'),raw.astype(np.uint8)*255)
            predictor.reset_image()
            cv2.imwrite(str(destination/f'{name}_ids.png'),ids)
            overlay=image.copy()
            for row in scores:
                selected=ids[...,0]==row['track_id']
                overlay[selected]=(.5*overlay[selected]+.5*np.array([255,120,0])).astype(np.uint8)
            for a in view['associations']:
                x0,y0,x1,y1=np.array(a['projected_box']).astype(int)
                color=(0,255,0) if a['match'] else (0,0,255)
                cv2.rectangle(overlay,(x0,y0),(x1,y1),color,2)
                cv2.putText(overlay,str(a['track_id']),(x0,max(20,y0)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            montage=np.concatenate([cv2.resize(x,(800,450)) for x in [image,overlay]],axis=1)
            cv2.imwrite(str(destination/f'{name}_comparison.jpg'),montage)
            row={'scene':scene,'view':name,'masks':scores,'dropped_unmatched_ids':[a['track_id'] for a in view['associations'] if a['match'] is None]}
            rows.append(row)
            (OUT/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(f'{scene} {name}: {len(scores)} visible detection masks',flush=True)
    result={'task_id':'VADGS-DETECTION-PROMPTED-SAM-GATE-20260926','device':'cpu','cpu_threads':2,'seed':0,
        'views':rows,'seconds':time.monotonic()-start,'raw_masks_clipped_to_3d_box':False,
        'training_inputs_modified':False,'dynamic_identity_block_marker_preserved':True,
        'scope':'development gate sidecars only; not full-scene mask qualification',
        'failure_ledger_refs':['V76-F02'],'failure_ledger_delta':'none; bounded fallback evidence'}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='views'}),flush=True)


if __name__=='__main__': main()
