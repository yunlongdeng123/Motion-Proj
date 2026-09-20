"""冻结ROI的配对图像响应与输入作用范围；与目标定位/科学失败结论分开。"""
import argparse
import json
import numpy as np
from common import RUN
from select_target import P1

def region(projections,frame,pad=16):
    boxes=[projections[k][frame]['bounds'] for k in ['clean','negative','positive'] if projections[k][frame] is not None]
    if not boxes:
        return None
    b=np.asarray(boxes)
    x0,y0=np.floor(b[:,:2].min(axis=0)-pad).astype(int)
    x1,y1=np.ceil(b[:,2:].max(axis=0)+pad).astype(int)
    x0,y0=max(0,x0),max(0,y0)
    x1,y1=min(1280,x1),min(704,y1)
    return None if x0>=x1 or y0>=y1 else tuple(map(int,(x0,y0,x1,y1)))

def input_audit():
    projections=json.loads((P1/'target_projections.json').read_text())
    clean=np.load(RUN/'conditions.npy',mmap_mode='r')
    result={}
    for name in ['negative','positive']:
        shifted=np.load(P1/f'conditions-{name}.npy',mmap_mode='r')
        rows=[]
        for f in range(54):
            roi=region(projections,f)
            assert roi is not None
            x0,y0,x1,y1=roi
            change=np.any(shifted[f]!=clean[f],axis=-1)
            n=int(change.sum())
            outside=n-int(change[y0:y1,x0:x1].sum())
            p=np.asarray(projections[name][f]['center_uv'])-projections['clean'][f]['center_uv']
            rows.append({'frame':f,'changed_pixels':n,'changed_pixels_outside_roi':outside,
                         'projection_delta_xy_px':p.tolist(),'roi':list(roi)})
        assert all(r['changed_pixels_outside_roi']==0 for r in rows)
        assert sum(r['changed_pixels'] for r in rows)>0
        result[name]={'changed_pixels_outside_roi':0,'frames':rows}
    (P1/'input_locality_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{'changed_pixels_outside_roi':v['changed_pixels_outside_roi'],
                         'mean_center_shift_px':float(np.mean([np.linalg.norm(r['projection_delta_xy_px']) for r in v['frames'][5:37]]))} for k,v in result.items()}),flush=True)

def analyze(seed,case):
    assert case!='clean'
    stem=f'{case}-seed{seed}'
    run=json.loads((P1/'rollouts'/f'{stem}.json').read_text())
    assert run['status']=='complete'
    clean=np.load(P1/'rollouts'/f'clean-seed{seed}.npy',mmap_mode='r')
    other=np.load(P1/'rollouts'/f'{stem}.npy',mmap_mode='r')
    projections=json.loads((P1/'target_projections.json').read_text())
    assert np.array_equal(clean[:5],other[:5]),'相同输入前缀不一致，禁止因果归因'
    rows=[]
    for f in range(237):
        delta=np.abs(other[f].astype(np.int16)-clean[f].astype(np.int16))
        roi=region(projections,f)
        roi_mean=None
        background_mean=float(delta.mean())
        if roi is not None:
            x0,y0,x1,y1=roi
            sub=delta[y0:y1,x0:x1]
            roi_mean=float(sub.mean())
            background_mean=float((delta.sum(dtype=np.int64)-sub.sum(dtype=np.int64))/(delta.size-sub.size))
        rows.append({'frame':f,'roi_mae_8bit':roi_mean,'background_mae_8bit':background_mean,
                     'whole_frame_mae_8bit':float(delta.mean()),'different_pixels':int(np.any(delta,axis=-1).sum())})
    windows={}
    for name,(lo,hi) in {'prefix':(0,4),'biased_window':(5,36),'recovery_window':(37,53),'late_no_target_claim':(54,236)}.items():
        sub=rows[lo:hi+1]
        windows[name]={'frames':[lo,hi], 'mean_roi_mae_8bit':float(np.mean([r['roi_mae_8bit'] for r in sub if r['roi_mae_8bit'] is not None])) if any(r['roi_mae_8bit'] is not None for r in sub) else None,
                       'mean_background_mae_8bit':float(np.mean([r['background_mae_8bit'] for r in sub])),
                       'mean_whole_frame_mae_8bit':float(np.mean([r['whole_frame_mae_8bit'] for r in sub]))}
    result={'case':case,'seed':seed,'prefix_raw_exactly_equal':True,'windows':windows,'frames':rows,
            'meaning':'raw uint8 image sensitivity relative to same-seed clean; not metric geometry error or amplification'}
    (P1/'rollouts'/f'{stem}.response.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='frames'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['inputs','response'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--case')
    a=p.parse_args()
    input_audit() if a.phase=='inputs' else analyze(a.seed,a.case)
