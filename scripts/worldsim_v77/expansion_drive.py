"""冻结规则的六例扩展；不对留出例补提示或调参数。"""
from pathlib import Path
import sys, os, time, signal, fcntl, traceback
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, '/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
from repair_common import read, dump
ROOT=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-EXPAND-EDIT-20260928/r1')

def alpha_for(base, i):
    load=lambda name: np.array(Image.open(base/name/f'{i:05}.png'))>0
    m,w,p=load('model_mask'),load('write_mask'),load('protect')
    t=np.clip(cv2.distanceTransform(m.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)/8,0,1)
    a=t*t*(3-2*t);a[w]=1;a[p]=0
    return m,w,p,a

def main():
    lock=open(ROOT/'expansion_drive.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'drive_state.json').exists()
    admission=dict(review='assistant source-image and mask review; not human verdict',human_verdict=None,
        admitted_primary_streams=[['processed_350',5],['processed_663',5],['processed_191',2],['processed_425',5],['processed_382',0]],
        unresolved_streams=[
            dict(scene='processed_756',camera=2,reason='Night target core absent in 22/30 frames; bus occlusion and propagation not separated. Mask input not admitted; heldout failure stays in denominator.'),
            dict(scene='processed_756',camera=0,reason='Only first two frame cores; edge/occlusion and tracking unresolved.'),
            dict(scene='processed_382',camera=2,reason='Only f29 core; target heavily occluded by other vehicles. No multi-view deletion claim.'),
            dict(scene='processed_191',camera=4,reason='Only f0 tiny image-edge core; no usable continuous cross-view.' )],
        limitation='Primary 191 target behind fence/tree and 663 small distant car remain deliberate stress cases; no claim of perfect masks or road legality.')
    dump(ROOT/'mask_admission.json',admission)
    from repair_drive import Engine,set_seed,torch
    torch.set_num_threads(4);cv2.setNumThreads(4)
    state=dict(state='loading',pid=os.getpid(),completed=[],human_verdict=None);dump(ROOT/'drive_state.json',state)
    start_all=time.monotonic()
    def timeout(*_):raise TimeoutError('One DriveEditor window exceeded 240 seconds')
    signal.signal(signal.SIGALRM,timeout)
    try:
        engine=Engine()
        for scene,cam in admission['admitted_primary_streams']:
            base=ROOT/scene/f'cam{cam}';out=base/'drive';out.mkdir()
            for folder in ['native','windows','composite']:(out/folder).mkdir()
            prev=None;written=set()
            for start in [0,9,18,27]:
                assert time.monotonic()-start_all<7200
                ids=[min(start+j,29) for j in range(10)];valid=min(10,30-start)
                engine.im=[np.array(Image.open(base/'rgb'/f'{i:05}.png').convert('RGB')) for i in ids]
                masks=[alpha_for(base,i) for i in ids];engine.masks=[v[0] for v in masks]
                engine.previous_segment_last_frame=prev;engine.im_result=[];set_seed(42)
                torch.cuda.reset_peak_memory_stats();beg=time.monotonic();signal.alarm(240)
                try:engine.predict(1,False,'Deletion')
                finally:signal.alarm(0)
                assert len(engine.im_result)==10
                wd=out/'windows'/f'{start:05}';wd.mkdir();checks=[]
                for j,(i,raw,im,parts) in enumerate(zip(ids,engine.im_result,engine.im,masks)):
                    m,w,p,a=parts
                    assert raw.dtype==np.uint8 and raw.shape==(576,1024,3)
                    comp=np.rint(raw*a[...,None]+im*(1-a[...,None])).astype('uint8')
                    checks.append(dict(frame=i,outside_changed=int(np.count_nonzero(comp[~m]!=im[~m])),protected_changed=int(np.count_nonzero(comp[p]!=im[p])),write_diff_native=int(np.count_nonzero(comp[w]!=raw[w]))))
                    assert checks[-1]['outside_changed']==checks[-1]['protected_changed']==checks[-1]['write_diff_native']==0
                    Image.fromarray(raw).save(wd/f'{j:02}.png')
                    if j<valid and i not in written:
                        Image.fromarray(raw).save(out/'native'/f'{i:05}.png');Image.fromarray(comp).save(out/'composite'/f'{i:05}.png');written.add(i)
                prev=np.array(Image.open(out/'composite'/f'{ids[valid-1]:05}.png'))
                row=dict(scene=scene,camera=cam,start=start,source_frames=ids,seconds=time.monotonic()-beg,peak_gib=torch.cuda.max_memory_allocated()/2**30,previous_condition=engine.used_previous_segment_condition,pixel_contracts=checks)
                state['completed'].append(row);state['state']='running';dump(ROOT/'drive_state.json',state);print('WINDOW_DONE',scene,cam,start,row['seconds'],flush=True)
            assert len(written)==30
        state.update(state='complete_pending_visual_review',elapsed_s=time.monotonic()-start_all);dump(ROOT/'drive_state.json',state)
    except Exception as ex:
        state.update(state='failed',error=repr(ex),trace=traceback.format_exc());dump(ROOT/'drive_state.json',state);raise

if __name__=='__main__':main()
