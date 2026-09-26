"""有界、可恢复的两路长上下文补景控制；同一run只允许一个执行器。"""
import datetime,fcntl,json,os,pathlib,signal,subprocess,time
ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R2-20260926/r1')
SCRIPT=str(next(p for p in [pathlib.Path(__file__).with_name('r2_long_context.py'),pathlib.Path(__file__).with_name('v77_r2_long_context.py')] if p.is_file()))
ROOT.mkdir(parents=True,exist_ok=True)
lock=(ROOT/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
registration={'task_id':'WS-V77-PIPELINE-R2-20260926','run_id':'r1','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_run':'WS-V77-EXPLICIT-POC-20260926/r1','source_commit':'e5bc729b','questions':['GT框无相交的候选是否碰到未标注静态障碍','恢复完整196帧上下文、放开40帧局部参考限制，能否改善原评价窗ProPainter残影'],'training_steps':0,'seed':7703,'models':'原SAM2.1 large与原ProPainter官方权重；不换生成器，不重跑Ω或Hunyuan','control':'原50/100帧RGB和mask使用源run符号链接冻结；只新增后续RGB与必要tail mask；同原帧比较，无目标背景GT','evaluation_frames':{'scene_0230':[0,5,10],'scene_0255':list(range(0,100,10))},'streams':{'scene_0230':{'camera':2,'old_frames':50,'new_frames':196},'scene_0255':{'camera':3,'old_frames':100,'new_frames':196}},'resource':'RTX3090 24GiB，GPU步骤串行，每阶段最多1800秒；CPU线程4','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'pending','human_verdict':None}
if not (ROOT/'registration.json').exists():(ROOT/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n')
steps=[('scene_0255','masks','worldsim-v77-sam2'),('scene_0230','paint','motionproj'),('scene_0255','paint','motionproj')]
state={'pid':os.getpid(),'state':'running','steps':[],'human_verdict':None}
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
for scene,action,environment in steps:
    cmd=[f'/root/autodl-tmp/envs/{environment}/bin/python',SCRIPT,action,'--scene',scene]
    state['current_step']={'scene':scene,'action':action};(ROOT/'controller_state.json').write_text(json.dumps(state,indent=2)+'\n')
    start=time.monotonic()
    with (ROOT/f'{scene}_{action}_controller.log').open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            code=proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            # Terminate the process group we created, including the nested
            # ProPainter child; killing only its wrapper would leak GPU work.
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=124
    state['steps'].append({'scene':scene,'action':action,'returncode':code,'elapsed_s':time.monotonic()-start})
    if code:
        state['state']='failed';break
else:state['state']='complete'
(ROOT/'controller_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
