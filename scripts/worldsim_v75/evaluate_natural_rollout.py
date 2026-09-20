"""真实未来先验证固定检测，再量化同一目标的配对生成偏差。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from evaluate_localization import model,predict,match
from prepare_argoverse import ROOT, CAMERA, crop_image

FRAMES=[0,15,30,45,60]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','evaluate'])
    parser.add_argument('--run-dir',required=True,type=Path);parser.add_argument('--case')
    a=parser.parse_args();out=a.run_dir;p=json.loads((out/'protocol.json').read_text());base=Path(p['base_dir'])
    m=model();projection=next(x for x in json.loads((base/'projections.json').read_text()) if x['id']==p['target'])['projections']
    if a.phase=='prepare':
        path=out/'evaluator_reference.json';assert not path.exists()
        bm=json.loads((base/'input_manifest.json').read_text());initial=np.asarray(Image.open(base/'initial_rgb.png'))
        b=projection[0]['bounds'];base_detection=match(predict(m,initial),b);cal=[]
        for dx in [-4,4]:
            shifted=np.zeros_like(initial)
            if dx>0:shifted[:,dx:]=initial[:,:-dx]
            else:shifted[:,:dx]=initial[:,-dx:]
            found=match(predict(m,shifted),(np.array(b)+[dx,0,dx,0]).tolist())
            err=None if found is None or base_detection is None else found['center'][0]-base_detection['center'][0]-dx
            cal.append({'dx':dx,'residual_px':err})
        images=sorted((ROOT/p['log_id']/'sensors/cameras'/CAMERA).glob('*.jpg'));ts=np.array([int(x.stem) for x in images],np.int64)
        times=np.load(base/'trajectory.npz')['timestamps_ns'];rows=[]
        for f in FRAMES:
            idx=int(np.argmin(abs(ts-times[f])));assert abs(ts[idx]-times[f])<=1000
            rgb=crop_image(images[idx],bm['crop_xyxy']);rgb.save(out/f'reference-{f:03d}.png')
            found=match(predict(m,np.asarray(rgb)),projection[f]['bounds'])
            rows.append({'frame':f,'match':found,'reference_bounds':projection[f]['bounds']})
        passed=all(x['residual_px'] is not None and abs(x['residual_px'])<=2 for x in cal) and all(x['match'] is not None for x in rows)
        result={'status':'passed' if passed else 'failed_stopped','calibration':cal,'frames':rows,
                'valid_matches':sum(x['match'] is not None for x in rows),'human_verdict':None,
                'interpretation':'real RGB evaluator admission; first two seconds only; not a reconstruction measurement'}
        path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        if not passed:raise RuntimeError('真实基线检测未通过，不启动生成')
        return
    path=out/a.case/'evaluation.json';assert not path.exists()
    ref=json.loads((out/'evaluator_reference.json').read_text());assert ref['status']=='passed'
    data=np.load(out/a.case/'clean.npy',mmap_mode='r');rows=[]
    for row in ref['frames']:
        f=row['frame'];found=match(predict(m,data[f]),row['reference_bounds'])
        distance=None if found is None else float(np.linalg.norm(np.array(found['center'])-row['match']['center']))
        rows.append({'frame':f,'match':found,'distance_to_real_center_px':distance})
    result={'status':'complete','frames':rows,'valid_matches':sum(x['match'] is not None for x in rows),
            'all_scheduled_frames':len(rows),'human_verdict':None,'metric':'2D center; not generated metric 3D state or policy consequence'}
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
