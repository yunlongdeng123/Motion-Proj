"""已有视频的有限光流交叉检查与普通投影对照，不新增生成。"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path('/root/autodl-tmp/motion_proj')
sys.path.insert(0,str(ROOT))
from motion_proj.auditor.flow_raft import RAFTFlow
from motion_proj.auditor.generated_tracks import _sample, _in_bounds

SOURCE=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-ROLLOUT-01')
OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-OBSERVATION-01/20260920-r1')
FRAMES=[0,15,30,45,60]
ARMS=['gt_clean','dvgt_metric','ordinary_bbox','reference_lidar']
RUNS={42:SOURCE/'20260920-r1',43:SOURCE/'20260920-seed43'}
WEIGHT=Path('/root/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')

def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def prepare():
    assert not OUT.exists(),'拒绝覆盖既有观测实验'
    assert WEIGHT.exists(),'只允许使用现有权重'
    OUT.mkdir(parents=True)
    p={'task_id':'WS-V75-OBSERVATION-01','run_id':'20260920-r1','frozen_utc':datetime.now(timezone.utc).isoformat(),
       'role':'exposed discovery-video observer cross-check; not independent scene confirmation',
       'frames':FRAMES,'seeds':[42,43],'arms':ARMS,'source_runs':{str(k):str(v) for k,v in RUNS.items()},
       'flow':'existing torchvision RAFT large C_T_SKHT_V2, FP32, 12 updates, full 704x1280 RGB [-1,1]',
       'source':'https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html',
       'queries':'Shi-Tomasi initial real RGB only, central 60% initial detector box, max64, quality0.01, minDistance3',
       'tracking':'first align real initial points to generated initial; then chain only the four fixed 0.5s intervals; no reinitialization, smoothing or future box association',
       'fb_max_error_px':2.,'minimum_common_points':8,'minimum_common_fraction':.25,
       'calibration':'real initial translated +/-4px horizontally; median point error <=2px, >=80% initial points retain FB consistency',
       'primary':'norm of coordinatewise median generated-minus-real displacement on the same initial point identities common to real and all four arms within each seed',
       'support':'all five scheduled times retained; full-window mean undefined if any time fails fixed support; point counts are not independent samples',
       'projection_control':'exact input cuboid projected bound-center shift relative to GT condition; vector comparison only, not a metric amplification factor',
       'stop':'one fixed observer/configuration only; no threshold or frame changes; any OOM stops; no new world-model generation, downloads or training',
       'boundary':'visible texture point correspondence is a 2D observation, not object center/metric 3D truth; occlusion and shape change can invalidate it',
       'human_verdict':None,'failure_ledger_refs':['V75-F01','V74-H2-F20','V74-H2-F21','V74-H2-F22']}
    write(OUT/'protocol.json',p)
    ref=json.loads((RUNS[42]/'evaluator_reference.json').read_text())
    rgb=np.array(Image.open(RUNS[42]/'reference-000.png'))
    box=np.array(ref['frames'][0]['match']['box']);c=(box[:2]+box[2:])/2;half=(box[2:]-box[:2])*.3
    mask=np.zeros(rgb.shape[:2],np.uint8);low=np.ceil(c-half).astype(int);high=np.floor(c+half).astype(int)
    mask[low[1]:high[1]+1,low[0]:high[0]+1]=255
    points=cv2.goodFeaturesToTrack(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),maxCorners=64,qualityLevel=.01,minDistance=3,mask=mask)
    assert points is not None and len(points)>=8
    points=points[:,0,:]
    write(OUT/'queries.json',{'initial_points_xy':points.tolist(),'count':len(points),'initial_box':box.tolist()})
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    draw.rectangle(box.tolist(),outline='yellow',width=2)
    for x,y in points:draw.ellipse((x-2,y-2,x+2,y+2),fill='#00e5b5')
    im.save(OUT/'query-overlay.png')
    print(json.dumps({'phase':'prepared','points':len(points),'run':str(OUT)}),flush=True)

def tensor(rgb):
    return torch.from_numpy(np.array(rgb,copy=True)).permute(2,0,1).float().unsqueeze(0)/127.5-1

def advance(raft,a,b,points,valid,path):
    # 复用已有RAFT封装；每次仅一对图，避免无关批处理显存增长。
    with torch.inference_mode():
        f=raft.flow(tensor(a),tensor(b))[0]
        back=raft.flow(tensor(b),tensor(a))[0]
        p=torch.from_numpy(points).to('cuda',torch.float32)
        forward=_sample(f,p);q=p+forward;reverse=_sample(back,q)
        residual=torch.linalg.vector_norm(forward+reverse,dim=-1)
        keep=_in_bounds(p,*a.shape[:2])&_in_bounds(q,*b.shape[:2])&(residual<=2.)
        q=q.cpu().numpy();error=residual.cpu().numpy();ok=valid&keep.cpu().numpy()
    # 保存点级原始观测和丢失原因；不只保存均值。
    np.savez(path,source=points,destination=q,fb_error=error,valid=ok)
    return q,ok,error

def run():
    assert (OUT/'protocol.json').exists() and not (OUT/'queue_result.json').exists()
    torch.set_num_threads(4);torch.manual_seed(0)
    queries=np.array(json.loads((OUT/'queries.json').read_text())['initial_points_xy'],np.float32)
    result={'status':'running','pid':os.getpid(),'completed':[],'human_verdict':None}
    write(OUT/'queue_result.json',result);started=time.monotonic()
    try:
        raft=RAFTFlow(device='cuda',dtype=torch.float32)
        initial=np.array(Image.open(RUNS[42]/'reference-000.png'))
        cal=[]
        for dx in [-4,4]:
            changed=np.zeros_like(initial)
            if dx>0:changed[:,dx:]=initial[:,:-dx]
            else:changed[:,:dx]=initial[:,-dx:]
            q,ok,fb=advance(raft,initial,changed,queries,np.ones(len(queries),bool),OUT/f'calibration-{dx}.npz')
            e=np.linalg.norm(q-queries-[dx,0],axis=1)
            cal.append({'dx':dx,'median_error_px':float(np.median(e)),'retained':int(ok.sum()),'total':len(ok)})
        passed=all(r['median_error_px']<=2 and r['retained']/r['total']>=.8 for r in cal)
        write(OUT/'calibration.json',{'status':'passed' if passed else 'failed','rows':cal})
        if not passed:raise RuntimeError('固定初帧平移校准未通过，停止此观察器')
        keys=[('real',None,None)]+[(f'seed{seed}-{arm}',seed,arm) for seed in [42,43] for arm in ARMS]
        for key,seed,arm in keys:
            folder=OUT/key;folder.mkdir()
            images=([np.array(Image.open(RUNS[42]/f'reference-{f:03d}.png')) for f in FRAMES] if seed is None
                    else [np.load(RUNS[seed]/arm/'clean.npy',mmap_mode='r')[f] for f in FRAMES])
            pts=queries.copy();valid=np.ones(len(pts),bool)
            if seed is not None:
                pts,valid,fb=advance(raft,initial,images[0],pts,valid,folder/'initial-alignment.npz')
            point_rows=[pts.copy()];valid_rows=[valid.copy()]
            for i in range(1,len(FRAMES)):
                pts,valid,fb=advance(raft,images[i-1],images[i],pts,valid,folder/f'step-{FRAMES[i]:03d}.npz')
                point_rows.append(pts.copy());valid_rows.append(valid.copy())
            np.savez(folder/'tracks.npz',points=np.array(point_rows),valid=np.array(valid_rows))
            row={'key':key,'support':[int(x.sum()) for x in valid_rows]}
            result['completed'].append(row);write(OUT/'queue_result.json',result)
            print(json.dumps(row),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result.update(wall_s=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write(OUT/'queue_result.json',result)

def summarize():
    assert json.loads((OUT/'queue_result.json').read_text())['status']=='complete'
    assert not (OUT/'result.json').exists()
    trace=json.loads((RUNS[43]/'trace/trace_data.json').read_text())
    projected={r['variant']:np.array([(np.array(p['bounds'][:2])+p['bounds'][2:])/2 for p in r['projections']]) for r in trace['states']}
    ref=np.load(OUT/'real/tracks.npz');n=ref['valid'].shape[1]
    rows=[]
    for seed in [42,43]:
        tracks={arm:np.load(OUT/f'seed{seed}-{arm}'/'tracks.npz') for arm in ARMS}
        common=ref['valid'].copy()
        for item in tracks.values():common &= item['valid']
        per=[]
        for i,f in enumerate(FRAMES):
            keep=common[i];count=int(keep.sum());admitted=count>=8 and count/n>=.25
            arms={}
            for arm in ARMS:
                d=tracks[arm]['points'][i]-ref['points'][i]
                dc=tracks[arm]['points'][i]-tracks['gt_clean']['points'][i]
                shift=projected[arm][i]-projected['gt_clean'][i]
                v=np.median(d[keep],axis=0) if admitted else None
                paired=np.median(dc[keep],axis=0) if admitted else None
                arms[arm]={'to_real_vector_px':None if v is None else v.tolist(),'to_real_distance_px':None if v is None else float(np.linalg.norm(v)),
                           'paired_clean_vector_px':None if paired is None else paired.tolist(),
                           'paired_clean_distance_px':None if paired is None else float(np.linalg.norm(paired)),
                           'projection_shift_px':shift.tolist(),'projection_distance_px':float(np.linalg.norm(shift)),
                           'point_error_p90_px':float(np.percentile(np.linalg.norm(d[keep],axis=1),90)) if admitted else None}
            per.append({'frame':f,'common_points':count,'total_initial_points':n,'admitted':admitted,'arms':arms})
        complete=all(r['admitted'] for r in per)
        means={arm:float(np.mean([r['arms'][arm]['to_real_distance_px'] for r in per])) if complete else None for arm in ARMS}
        rows.append({'seed':seed,'full_window_measurable':complete,'means_px':means,'frames':per})
    result={'status':'complete','rows':rows,'human_verdict':None,'failure_ledger_delta':'none',
            'world_model_generation_calls':0,'source_logs':1,'boundary':'texture point correspondence, not object centroid or metric 3D; no independent scene confirmation'}
    write(OUT/'result.json',result)
    print(json.dumps({'status':'complete','rows':[{k:v for k,v in row.items() if k!='frames'} for row in rows],
                      'common_support':[[r['common_points'] for r in row['frames']] for row in rows]}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','summarize'])
    phase=parser.parse_args().phase
    {'prepare':prepare,'run':run,'summarize':summarize}[phase]()
