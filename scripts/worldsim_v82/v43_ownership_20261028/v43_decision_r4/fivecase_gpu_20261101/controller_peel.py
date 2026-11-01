"""R4显露来源控制，五例、固定规则；不读评分，不训练模型。"""
import concurrent.futures,json,pathlib,time
import controller as c
R=c.ROOT
def run(case):
    # 前置只有完整原位贡献记录与冻结动态子集，不等待编辑评分。
    cpu=R/'cpu'/case
    while not (cpu/'dynamic_binding.json').exists() or not (R/'cpu_refined'/case/'tasks.json').exists():time.sleep(2)
    out=R/'gpu_exposed'/case
    c.step(case,'peel',cpu,out/'source_peel49')
    source=R/'cpu_exposed'/case
    c.step(case,'smoke',source,out/'exposed_smoke')
    c.step(case,'enumerate',source,out/'exposed_enum',out/'exposed_smoke/inference_state.pt')
    c.step(case,'sequence',source,out/'exposed_sequence',out/'exposed_smoke/inference_state.pt')
    return case
if __name__=='__main__':
    # 基础两域队列结束后再进入下一队列，避免增加并发进程。
    files=['controller_final.json','controller_final_gpu_r2.json','controller_final_gpu_564_replay.json']
    while not all((R/f).exists() for f in files):time.sleep(2)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run,x):x for x in c.CASES}
        for f in concurrent.futures.as_completed(jobs):
            try:f.result();results.append(dict(case=jobs[f],status='COMPLETE'))
            except Exception as e:results.append(dict(case=jobs[f],status='FAILED_RETAINED',error=str(e)))
    (R/'controller_final_exposed.json').write_text(json.dumps(dict(results=results,training_updates=0),indent=2))
