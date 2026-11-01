"""完成完整宽域端点视频，并精确补跑受OOM影响的唯一阶段。"""
import concurrent.futures,json,pathlib,time
import controller as c
R=c.ROOT
RUNS={'waymo-train-008_a37':'gpu','waymo-train-212_a0':'gpu_r2','waymo-train-212_a35':'gpu_r2',
      'waymo-train-564_a0':'gpu_564_replay','waymo-train-564_a22':'gpu_564_replay'}
def run(case):
    old=R/RUNS[case]/case
    if case=='waymo-train-212_a35' and not (old/'refined_sequence/gpu_manifest.json').exists():
        failed=old/'refined_sequence'
        if failed.exists():failed.rename(old/'refined_sequence_OOM_failed')
        c.step(case,'sequence',R/'cpu_refined'/case,old/'refined_sequence',old/'refined_smoke/inference_state.pt')
    c.step(case,'sequence',R/'cpu'/case,R/'gpu_broad_sequence'/case/'broad_sequence',old/'parent_smoke/inference_state.pt')
    return case
if __name__=='__main__':
    while not (R/'controller_final_exposed.json').is_file():time.sleep(2)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run,k):k for k in c.CASES}
        for f in concurrent.futures.as_completed(jobs):
            try:f.result();results.append(dict(case=jobs[f],status='COMPLETE'))
            except Exception as e:results.append(dict(case=jobs[f],status='FAILED_RETAINED',error=str(e)))
    (R/'controller_final_controls.json').write_text(json.dumps(dict(results=results,max_parallel_workers=2,training_updates=0),indent=2))
