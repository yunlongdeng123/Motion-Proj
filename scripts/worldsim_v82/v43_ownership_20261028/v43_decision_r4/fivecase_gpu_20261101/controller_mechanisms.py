"""用户补充的机制隔离核查；真实证据和独立E1控制分开。"""
import concurrent.futures,json,pathlib,subprocess,sys,time
R=pathlib.Path(__file__).resolve().parent
CASES=['waymo-train-212_a0','waymo-train-564_a0','waymo-train-212_a35','waymo-train-564_a22','waymo-train-008_a37']
def run(case):
    with (R/('mechanism_gpu_'+case+'.log')).open('w') as f:
        p=subprocess.run(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(R/'mechanism_gpu.py'),'--case',case],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(case)
    return case
if __name__=='__main__':
    while not (R/'controller_final_fullsource.json').exists():time.sleep(2)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run,c):c for c in CASES}
        for f in concurrent.futures.as_completed(jobs):
            try:f.result();results.append(dict(case=jobs[f],status='COMPLETE'))
            except Exception as e:results.append(dict(case=jobs[f],status='FAILED_RETAINED',error=str(e)))
    with (R/'sim_factor_controls.log').open('w') as f:
        p=subprocess.run(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(R/'sim_factor_controls.py')],stdout=f,stderr=subprocess.STDOUT)
    (R/'mechanism_gpu_final.json').write_text(json.dumps(dict(real_cases=results,sim_control_exit_code=p.returncode,training_labels=0),indent=2))
