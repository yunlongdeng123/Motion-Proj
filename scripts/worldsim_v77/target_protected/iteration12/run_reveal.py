"""有界数据控制器；等待已运行SAM，4路CPU规划，再渲染候选。不会启动训练。"""
from pathlib import Path
import os, json, sys, time, subprocess, fcntl
O = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r23')
S = Path(__file__).parent
PY = '/root/autodl-tmp/envs/motionproj/bin/python'


def save(d):
    p = O/'data_controller.json'; tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n'); tmp.replace(p)


def main():
    lock = open(O/'data_controller.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = {'stage': 'waiting_existing_SAM2', 'pid': os.getpid(), 'children': [], 'training_steps': 0}
    save(state); deadline = time.time()+1800
    try:
        while True:
            p = O/'segmentation_state.json'
            if p.exists():
                sam = json.loads(p.read_text())
                if sam['stage'] == 'complete_quarantined_pending_synthetic_QA': break
                if sam['stage'] == 'failed': raise RuntimeError(sam.get('error'))
                os.kill(sam['pid'], 0)
            if time.time() > deadline: raise TimeoutError('等待既有SAM超过30分钟，需检查')
            time.sleep(10)
        env = os.environ | {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
        import reveal_factory
        reveal_factory.register()
        jobs = []
        for shard in range(4):
            log = open(O/f'lane_{shard}.log', 'a')
            p = subprocess.Popen([PY, str(S/'reveal_factory.py'), '--shard', str(shard), '--count', '4'], stdout=log, stderr=subprocess.STDOUT, env=env)
            jobs.append(p)
        state.update(stage='CPU_lane_planning', children=[p.pid for p in jobs]); save(state)
        codes = [p.wait() for p in jobs]
        if any(codes): raise RuntimeError('lane worker failure: '+repr(codes))
        log = open(O/'render_reveal.log', 'a')
        p = subprocess.Popen([PY, str(S/'render_reveal.py')], stdout=log, stderr=subprocess.STDOUT, env=env)
        state.update(stage='rendering_quarantined_pilot', children=[p.pid]); save(state)
        if p.wait(): raise RuntimeError('render worker failure; see render_reveal.log')
        state.update(stage='complete_pending_independent_QA', children=[]); save(state)
    except Exception as e:
        state.update(stage='engineering_error', error=repr(e)); save(state); raise


if __name__ == '__main__': main()
