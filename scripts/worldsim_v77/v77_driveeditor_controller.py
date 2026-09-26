"""有界串行GPU控制；分别启动paint/drive阶段，不自动重试失败。"""
import argparse, fcntl, json, os, pathlib, subprocess, sys, time
from v77_driveeditor_compare import ROOT, dump, SCENES

p=argparse.ArgumentParser();p.add_argument('stage',choices=['paint','drive']);a=p.parse_args()
ROOT.mkdir(parents=True,exist_ok=True)
lock=(ROOT/'gpu_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
statefile=ROOT/f'{a.stage}_state.json'
if statefile.exists():raise RuntimeError('控制器状态已存在；先检查，不能重复启动或覆盖')
if a.stage=='drive':
    assert json.loads((ROOT/'adapter_validation.json').read_text())['status']=='passed'
    assert pathlib.Path('/root/autodl-tmp/external/worldsim_v75_downstream_bench/DriveEditor/checkpoints/model.safetensors').is_file()
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',DRIVEEDITOR_SEQUENTIAL_CFG='1',PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128')
state={'pid':os.getpid(),'state':'running','steps':[],'human_verdict':None}
stages=[(s,v) for s in SCENES for v in ('A','B')] if a.stage=='paint' else [(s,'C') for s in SCENES]
for scene,variant in stages:
    environment='motionproj' if a.stage=='paint' else 'driveeditor'
    cmd=[f'/root/autodl-tmp/envs/{environment}/bin/python',str(pathlib.Path(__file__).with_name('v77_driveeditor_compare.py')),a.stage,'--scene',scene]
    if a.stage=='paint':cmd+=['--variant',variant]
    state['current']={'scene':scene,'variant':variant};dump(statefile,state)
    t=time.monotonic()
    with (ROOT/f'{scene}_{variant}.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:code=proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    state['steps'].append({'scene':scene,'variant':variant,'returncode':code,'elapsed_s':time.monotonic()-t})
    if code:state['state']='failed';break
else:state['state']='complete'
dump(statefile,state);print(json.dumps(state),flush=True)
if state['state']!='complete':raise SystemExit(state['steps'][-1]['returncode'] or 1)
if a.stage=='drive' and state['state']=='complete':
    with (ROOT/'export.log').open('w') as log:
        code=subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('v77_driveeditor_export.py'))],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600).returncode
    if code:
        dump(ROOT/'review_export.json',{'state':'failed','returncode':code,'model_outputs_preserved':True})
        raise SystemExit(code)
