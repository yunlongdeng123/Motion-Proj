"""r26：仅增加可见保护输入规则，统一八个真实DEV的原模型/r7。"""
from pathlib import Path
import os,sys,json,time,gc,fcntl,signal
os.environ['DRIVEEDITOR_SEQUENTIAL_CFG']='1';os.environ['OMP_NUM_THREADS']='4'
S=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77');sys.path[:0]=[str(S)]
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929');O=T/'r26'
import numpy as np
from PIL import Image


def read(p):return json.loads(p.read_text())
def dump(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');q.replace(p)


def main():
    lock=open(O/'baseline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cases=read(O/'real_input_plan.json')['cases']
    assert read(O/'assistant_input_review.json')['approved_for_bounded_inference'] is True
    plan={'role':'engineering_control_new_input_not_new_model','cases':[c['eval_id'] for c in cases],
          'arms':['base','r7'],'resolution':[576,1024],'frames':10,'seed':42,'steps':25,
          'mask_policy':'visible_protection_v1','max_new_windows':16,'previous_condition':False,'human_verdict':None}
    if not (O/'baseline_plan.json').exists():dump(O/'baseline_plan.json',plan)
    else:assert read(O/'baseline_plan.json')==plan
    dest=O/'baseline';dest.mkdir(exist_ok=True)
    state=read(O/'baseline_state.json') if (O/'baseline_state.json').exists() else {'completed':[]}
    done={(r['case'],r['arm']) for r in state['completed']}
    from repair_drive import Engine,set_seed,torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);e=None
    def timeout(*_):raise TimeoutError('入口修复单窗限300秒')
    signal.signal(signal.SIGALRM,timeout)
    for arm in plan['arms']:
        if all((c['eval_id'],arm) in done for c in cases):continue
        if e is not None:del e;gc.collect();torch.cuda.empty_cache()
        state.update(stage='loading_'+arm,pid=os.getpid());dump(O/'baseline_state.json',state);e=Engine()
        if arm=='r7':
            patch=load_file(str(T/'r7/encoder_fixed_lowres/training/attention_step_0160.safetensors'))
            assert len(patch)==80
            _,extra=e.model.load_state_dict(patch,strict=False);assert not extra;del patch
        for c in cases:
            cid=c['eval_id']
            if (cid,arm) in done:continue
            f=Path(c['folder']);ids=c['frames'];out=dest/cid
            for role in ['input','mask','condition',arm,arm+'_native']:(out/role).mkdir(parents=True,exist_ok=True)
            rgb=[np.asarray(Image.open(f/'rgb'/f'{i:05}.jpg').convert('RGB')) for i in ids]
            masks=[np.asarray(Image.open(f/'model_mask'/f'{i:05}.png'))>0 for i in ids]
            alphas=[np.asarray(Image.open(f/'alpha'/f'{i:05}.png')).astype('float32')/255 for i in ids]
            for j,(im,h) in enumerate(zip(rgb,masks)):
                cond=im.copy();cond[h]=127
                for role,arr in [('input',im),('mask',h.astype('uint8')*255),('condition',cond)]:Image.fromarray(arr).save(out/role/f'{j:05}.png')
            state.update(stage='inference',current_case=cid,current_arm=arm);dump(O/'baseline_state.json',state)
            e.im=rgb;e.masks=masks;e.masked_condition=None;e.im_result=[];e.previous_segment_last_frame=None
            set_seed(42);start=time.monotonic();signal.alarm(300)
            try:e.predict(1,False,'Deletion')
            finally:signal.alarm(0)
            assert len(e.im_result)==10
            for j,(raw,im,h,alpha) in enumerate(zip(e.im_result,rgb,masks,alphas)):
                comp=np.rint(raw*alpha[...,None]+im*(1-alpha[...,None])).clip(0,255).astype('uint8')
                assert np.array_equal(comp[~h],im[~h])
                if not h.any():assert np.array_equal(comp,im)
                Image.fromarray(raw).save(out/(arm+'_native')/f'{j:05}.png');Image.fromarray(comp).save(out/arm/f'{j:05}.png')
            rec={'case':cid,'arm':arm,'seconds':time.monotonic()-start,'old_comparison':str(T/'r21/baseline'/cid/arm),'human_verdict':None}
            state['completed'].append(rec);done.add((cid,arm));dump(O/'baseline_state.json',state);print('INPUT_FIX',rec,flush=True)
    state.update(stage='complete_pending_review',new_windows=16);dump(O/'baseline_state.json',state)


if __name__=='__main__':main()
