"""九例相同DriveEditor条件，逐窗保存；输入失败也单列诊断输出。"""
from nine_common import *
import os,time,signal,fcntl,traceback,numpy as np,cv2
from PIL import Image
lock=open(ROOT/'drive.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert read(ROOT/'mask_state.json')['state']=='complete'
assert not (ROOT/'drive_state.json').exists()
reg=read(ROOT/'registration.json');state=dict(state='loading',pid=os.getpid(),completed=[],human_verdict=None);dump(ROOT/'drive_state.json',state)
from repair_drive import Engine,set_seed,torch
torch.set_num_threads(4);cv2.setNumThreads(4);engine=Engine();beg=time.monotonic()
def deadline(*_):raise TimeoutError('DriveEditor单窗超过240秒')
signal.signal(signal.SIGALRM,deadline)
try:
    for s in reg['scenes']:
        for v in s['streams']:
            c=v['camera'];base=ROOT/s['name']/f'cam{c}';out=base/'background';out.mkdir()
            if not v['active']:
                for f in range(30):(out/f'{f:05}.png').symlink_to(base/'rgb'/f'{f:05}.png')
                continue
            native=base/'native';native.mkdir();wins=base/'windows';wins.mkdir();prev=None;written=set()
            for start in [0,9,18,27]:
                ids=[min(start+j,29) for j in range(10)];valid=min(10,30-start)
                engine.im=[np.array(Image.open(base/'rgb'/f'{i:05}.png').convert('RGB')) for i in ids]
                engine.masks=[np.array(Image.open(base/'model_mask'/f'{i:05}.png'))>0 for i in ids]
                engine.previous_segment_last_frame=prev;engine.im_result=[];set_seed(42)
                has_hole=any(m.any() for m in engine.masks);start_time=time.monotonic();torch.cuda.reset_peak_memory_stats()
                if has_hole:
                    signal.alarm(240)
                    try:engine.predict(1,False,'Deletion')
                    finally:signal.alarm(0)
                    assert len(engine.im_result)==10
                else:engine.im_result=[im.copy() for im in engine.im]
                wd=wins/f'{start:05}';wd.mkdir();checks=[]
                for j,(i,im,raw,m) in enumerate(zip(ids,engine.im,engine.im_result,engine.masks)):
                    write=np.array(Image.open(base/'write_mask'/f'{i:05}.png'))>0;protect=np.array(Image.open(base/'protect'/f'{i:05}.png'))>0
                    dist=cv2.distanceTransform(m.astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);t=np.clip(dist/8,0,1);alpha=t*t*(3-2*t);alpha[write]=1;alpha[protect]=0
                    comp=np.rint(raw*alpha[...,None]+im*(1-alpha[...,None])).astype('uint8')
                    check=dict(frame=i,outside_changed=int(np.count_nonzero(comp[~m]!=im[~m])),protected_changed=int(np.count_nonzero(comp[protect]!=im[protect])),write_diff_native=int(np.count_nonzero(comp[write]!=raw[write])))
                    assert check['outside_changed']==check['protected_changed']==check['write_diff_native']==0;checks.append(check)
                    Image.fromarray(raw).save(wd/f'{j:02}.png')
                    if j<valid and i not in written:
                        Image.fromarray(comp).save(out/f'{i:05}.png');Image.fromarray(raw).save(native/f'{i:05}.png');written.add(i)
                prev=np.array(Image.open(out/f'{ids[valid-1]:05}.png'))
                row=dict(scene=s['name'],camera=c,start=start,source_frames=ids,generated=has_hole,skip_reason=None if has_hole else 'all_model_masks_empty_RGB_unchanged',previous_condition=bool(engine.used_previous_segment_condition) if has_hole else False,seconds=time.monotonic()-start_time,peak_gib=torch.cuda.max_memory_allocated()/2**30,pixel_contracts=checks)
                state['completed'].append(row);state['state']='running';dump(ROOT/'drive_state.json',state);progress('drive',scene=s['name'],camera=c,start=start,generated=has_hole,windows_done=len(state['completed']))
            assert len(written)==30
    state.update(state='complete',seconds=time.monotonic()-beg);dump(ROOT/'drive_state.json',state);progress('drive',state='complete',windows=len(state['completed']))
except Exception as ex:
    state.update(state='failed_engineering',error=repr(ex),trace=traceback.format_exc());dump(ROOT/'drive_state.json',state);raise
