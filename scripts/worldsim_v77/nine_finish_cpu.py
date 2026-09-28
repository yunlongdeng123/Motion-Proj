"""一次性的既定QUERY/验证/导出队列，等待本次模型和渲染产物。"""
from nine_common import *
import time,subprocess,os,fcntl,traceback
lock=open(ROOT/'finish.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'finish_state.json').exists()
reg=read(ROOT/'registration.json');state=dict(state='waiting_for_models_and_layers',pid=os.getpid(),queried=[],human_verdict=None);dump(ROOT/'finish_state.json',state);beg=time.time();env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='');py='/root/autodl-tmp/envs/motionproj/bin/python'
def run(args,log):
    with (ROOT/log).open('w') as fp:subprocess.run([py,*args],cwd='/root/autodl-tmp/work',env=env,stdout=fp,stderr=subprocess.STDOUT,check=True)
try:
    while len(state['queried'])<9:
        assert time.time()-beg<14400,'本次有界运行等待超过4小时'
        seq=read(ROOT/'sequence_state.json')
        if seq['state']=='failed_engineering':raise RuntimeError(seq)
        for s in reg['scenes']:
            name=s['name'];base=ROOT/name
            if name in state['queried']:continue
            if (base/'background_world/029/summary.json').exists() and (base/'actor_layers/render_summary.json').exists():
                state.update(state='query',current=name);dump(ROOT/'finish_state.json',state);run(['nine_query.py','--scene',name],f'query_{name}.log');state['queried'].append(name);dump(ROOT/'finish_state.json',state);print('QUERY_FINISHED',name,flush=True)
        if len(state['queried'])<9:time.sleep(20)
    while read(ROOT/'sequence_state.json')['state']!='complete':time.sleep(5)
    state.update(state='validate',current=None);dump(ROOT/'finish_state.json',state);run(['nine_validate.py'],'validation.log')
    state['state']='export';dump(ROOT/'finish_state.json',state);run(['nine_export_review.py'],'export.log')
    state.update(state='complete',seconds=time.time()-beg,bundle=str(ROOT/'review_bundle.tar'));dump(ROOT/'finish_state.json',state);print('NINE_FINISHED',flush=True)
except Exception as exc:
    state.update(state='failed_engineering',error=repr(exc),trace=traceback.format_exc());dump(ROOT/'finish_state.json',state);raise
