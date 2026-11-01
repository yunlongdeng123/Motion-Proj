import concurrent.futures,json,pathlib,subprocess,sys,time
import controller as c
R=c.ROOT
def run(case):
    cpu=R/'cpu_fullsource'/case;out=R/'gpu_fullsource'/case
    c.step(case,'smoke',cpu,out/'source_smoke')
    c.step(case,'enumerate',cpu,out/'source_enum',out/'source_smoke/inference_state.pt')
    c.step(case,'sequence',cpu,out/'source_sequence',out/'source_smoke/inference_state.pt')
    return case
if __name__=='__main__':
    # 只读CPU来源准备可与末尾GPU控制并行。
    while not all((R/'gpu_exposed'/x/'source_peel49/summary.json').exists() for x in c.CASES):time.sleep(2)
    with (R/'prepare_fullsource.log').open('w') as f:
        p=subprocess.run([sys.executable,str(R/'prepare_fullsource.py')],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError('完整来源候选准备失败')
    while not (R/'controller_final_controls.json').exists():time.sleep(2)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run,k):k for k in c.CASES}
        for f in concurrent.futures.as_completed(jobs):
            try:f.result();results.append(dict(case=jobs[f],status='COMPLETE'))
            except Exception as e:results.append(dict(case=jobs[f],status='FAILED_RETAINED',error=str(e)))
    (R/'controller_final_fullsource.json').write_text(json.dumps(dict(results=results,max_parallel_workers=2,training_updates=0),indent=2))
