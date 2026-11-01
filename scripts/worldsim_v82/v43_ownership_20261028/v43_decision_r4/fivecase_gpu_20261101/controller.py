"""已授权五例的有限队列；每例新进程恢复，不自动扩大或训练。"""
import concurrent.futures,json,os,pathlib,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parent
PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
CASES=['waymo-train-212_a0','waymo-train-564_a0','waymo-train-212_a35','waymo-train-564_a22','waymo-train-008_a37']
if os.environ.get('FIVE_CASES'):CASES=os.environ['FIVE_CASES'].split(',')
events=[]
def event(x):
    x=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**x);events.append(x)
    with (ROOT/'controller_events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    print(json.dumps(x),flush=True)
def step(case,phase,cpu,out,resume=None):
    cmd=[PY,str(ROOT/'run_five.py'),'--cpu',str(cpu),'--output',str(out),'--phase',phase]
    if resume:cmd+=['--resume',str(resume)]
    log=ROOT/'logs'/(case+'_'+out.name+'.log');log.parent.mkdir(exist_ok=True)
    event(dict(case=case,phase=phase,output=str(out),state='START',command=cmd))
    with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
    event(dict(case=case,phase=phase,output=str(out),state='PASS' if r.returncode==0 else 'FAIL',exit_code=r.returncode,log=str(log)))
    if r.returncode:raise RuntimeError(str(log))
def case_run(case):
    cpu=ROOT/'cpu'/case;out=ROOT/os.environ.get('FIVE_GPU_OUTPUT','gpu')/case
    if not (cpu/'tasks.json').is_file():raise ValueError('CPU合同尚未生成:'+case)
    step(case,'audit',cpu,out/'audit49')
    step(case,'smoke',cpu,out/'parent_smoke')
    step(case,'enumerate',cpu,out/'parent_enum',out/'parent_smoke/inference_state.pt')
    refined=ROOT/'cpu_refined'/case
    step(case,'smoke',refined,out/'refined_smoke')
    step(case,'enumerate',refined,out/'refined_enum',out/'refined_smoke/inference_state.pt')
    step(case,'sequence',refined,out/'refined_sequence',out/'refined_smoke/inference_state.pt')
    return case
if __name__=='__main__':
    results=[]
    # 32GB卡，212已测峰值3.27GB；最多两例，各自保留完整来源和输出。
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(case_run,c):c for c in CASES}
        for f in concurrent.futures.as_completed(jobs):
            c=jobs[f]
            try:f.result();results.append(dict(case=c,status='COMPLETE'))
            except Exception as e:results.append(dict(case=c,status='FAILED_RETAINED',error=str(e)))
    final='controller_final.json' if os.environ.get('FIVE_GPU_OUTPUT','gpu')=='gpu' else 'controller_final_'+os.environ['FIVE_GPU_OUTPUT']+'.json'
    (ROOT/final).write_text(json.dumps(dict(results=results,automatic_training=False,automatic_shutdown=False),indent=2))
    event(dict(state='CONTROLLER_FINISHED',results=results))
