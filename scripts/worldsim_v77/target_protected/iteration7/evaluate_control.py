"""同一固定六例，比较原模型/r6/原尺寸空间臂/原尺寸扩大更新臂。"""
from pathlib import Path
import argparse, json, sys, time, shutil, fcntl, signal
import numpy as np
from PIL import Image
TASK=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
ROOT=TASK/'r7'
REPO=Path('/root/autodl-tmp/motion_proj_v77')
sys.path.insert(0,str(REPO/'scripts/worldsim_v77'))
sys.path.insert(0,str(Path(__file__).parent))
from audit import protected_mask, dump

def inputs(c):
    folder=Path(c['folder'])
    load=lambda role,rgb:[np.asarray(Image.open(folder/role/f'{i:03}.png').convert('RGB' if rgb else 'L')) for i in c['frames']]
    y=load('Y',True);x=load('X',True);h=[v>0 for v in load('model_hole',False)]
    b=[protected_mask(folder,i,h[0].shape,c['type']) for i in c['frames']]
    assert all(v.any() for v in h)
    return x,y,h,b

def metrics(im,y,h,b):
    error=(im.astype('float32')-y.astype('float32'))/255
    row={}
    for name,r in [('hole',h),('protected_inside_hole',b&h),('native_outside_hole',~h)]:
        if not r.any():row[name]=None;continue
        mse=float((error[r]**2).mean());row[name]={'pixels':int(r.sum()),'MAE':float(abs(error[r]).mean()),'PSNR':float(-10*np.log10(max(mse,1e-12)))}
    return row

def main(a):
    from repair_drive import Engine,set_seed,torch
    from safetensors.torch import load_file
    plan=json.loads((ROOT/'evaluation_plan.json').read_text());dest=ROOT/'evaluation';dest.mkdir(exist_ok=True)
    lock=open(dest/'evaluate.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((dest/'state.json').read_text()) if (dest/'state.json').exists() else {'completed':[],'stage':'starting'}
    done={(r['eval_id'],r['arm']) for r in state['completed']}
    cases=plan['cases'];arms=a.arms
    e=None
    def timeout(*_):raise TimeoutError('fixed sampling window >300sec')
    signal.signal(signal.SIGALRM,timeout);torch.set_num_threads(4)
    for arm in arms:
        required={(c['eval_id'],arm) for c in cases}
        if required<=done:continue
        if arm in ('encoder_fixed_lowres','encoder_fixed_native'):
            status=json.loads((ROOT/arm/'training/state.json').read_text());assert status['stage']=='complete' and status['steps']==160
        # 新Engine恢复原权重，防止r6/上一控制臂未覆盖的参数污染。
        if e is not None:del e;torch.cuda.empty_cache()
        e=Engine()
        if arm!='base':
            patch_path=(TASK/'r6' if arm=='r6' else ROOT/arm)/'training/attention_step_0160.safetensors'
            patch=load_file(str(patch_path));config=json.loads((patch_path.parent/'config.json').read_text())
            assert set(patch)==set(config['trainable_tensors'])
            _,unexpected=e.model.load_state_dict(patch,strict=False);assert not unexpected
            dump(dest/f'{arm}_patch_loaded.json',{'path':str(patch_path),'loaded_tensors':len(patch),'architecture_unchanged':True})
            del patch
        for c in cases:
            if (c['eval_id'],arm) in done:continue
            folder=dest/c['eval_id'];folder.mkdir(exist_ok=True)
            x,y,h,b=inputs(c)
            for role in ['GT','input','condition',arm,arm+'_native']:(folder/role).mkdir(exist_ok=True)
            if arm=='base':
                for i,(xx,yy,hh) in enumerate(zip(x,y,h)):
                    Image.fromarray(yy).save(folder/'GT'/f'{i:05}.png');Image.fromarray(xx).save(folder/'input'/f'{i:05}.png')
                    cc=xx.copy();cc[hh]=127;Image.fromarray(cc).save(folder/'condition'/f'{i:05}.png')
            # 既有四个留出原/微调窗条件完全相同，明确复制来源，避免重复采样。
            previous=TASK/'r6/evaluation'/('synthetic_'+c['case_id'])
            reused=c['role']=='heldout' and arm in ('base','r6')
            start=time.monotonic();scores=[]
            if reused:
                for i,(xx,yy,hh) in enumerate(zip(x,y,h)):
                    assert np.array_equal(xx,np.asarray(Image.open(previous/'input'/f'{i:05}.png')))
                    assert np.array_equal(yy,np.asarray(Image.open(previous/'GT'/f'{i:05}.png')))
                    assert np.array_equal(hh,np.asarray(Image.open(previous/'mask'/f'{i:05}.png'))>0)
                role='base' if arm=='base' else 'finetuned'
                raws=[np.asarray(Image.open(previous/(role+'_native')/f'{i:05}.png')) for i in range(10)]
            else:
                e.im=x;e.masks=h;e.previous_segment_last_frame=None;e.im_result=[];set_seed(42)
                torch.cuda.reset_peak_memory_stats();signal.alarm(300)
                try:e.predict(1,False,'Deletion')
                finally:signal.alarm(0)
                raws=e.im_result;assert len(raws)==10
            for i,(xx,yy,hh,bb,raw) in enumerate(zip(x,y,h,b,raws)):
                comp=np.where(hh[...,None],raw,xx)
                assert not np.any(comp[~hh]!=yy[~hh])
                Image.fromarray(comp).save(folder/arm/f'{i:05}.png');Image.fromarray(raw).save(folder/(arm+'_native')/f'{i:05}.png')
                scores.append(dict(frame=i,**metrics(raw,yy,hh,bb)))
            row={'eval_id':c['eval_id'],'arm':arm,'role':c['role'],'source':c,'scores':scores,'seconds':time.monotonic()-start,
                 'reused_from':str(previous) if reused else None,'seed':42,'sampling_steps':25,
                 'peak_allocated_GiB':None if reused else torch.cuda.max_memory_allocated()/2**30}
            dump(folder/f'{arm}_metrics.json',row);state['completed'].append({k:row[k] for k in ('eval_id','arm','seconds','reused_from')});done.add((c['eval_id'],arm))
            state['stage']=arm;dump(dest/'state.json',state);print(json.dumps(state['completed'][-1]),flush=True)
    state['stage']='complete_requested_arms';state['arms_requested']=arms;dump(dest/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',choices=['base','r6','encoder_fixed_lowres','encoder_fixed_native'],required=True);main(p.parse_args())
