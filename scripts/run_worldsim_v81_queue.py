"""单卡串行消费冻结队列；两模型分别启动独立进程，无自动恢复。"""
import argparse,json,os,time,traceback
from pathlib import Path
import run_worldsim_v81_inference as inference

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['dvgt','vggt']);p.add_argument('--queue',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    jobs=[json.loads(l) for l in Path(a.queue).read_text().splitlines()];jobs=[j for j in jobs if j['method']==a.method]
    progress=out/(a.method+'_queue_progress.jsonl');started=time.time()
    for j in jobs:
        path=out/a.method/j['window_id']/j.get('output_key',j['variant'])
        if (path/'result.json').exists():continue
        argv=['--method',a.method,'--manifest',j['manifest'],'--variant',j['variant'],'--out',str(path),'--execute']
        if j.get('anchor_camera'):argv+=['--anchor-camera',j['anchor_camera']]
        if j.get('texture_case'):argv+=['--texture-case',j['texture_case']]
        t=time.time();record={'job':j,'out':str(path),'started':t}
        try:
            inference.main(argv);record.update(status='DONE',wall_s=time.time()-t)
        except Exception as e:
            record.update(status='ERROR',error=str(e),traceback=traceback.format_exc());path.mkdir(parents=True,exist_ok=True)
            (path/'error.json').write_text(json.dumps(record,indent=2));print(record['traceback'],flush=True)
            with progress.open('a') as f:f.write(json.dumps(record)+'\n')
            # 不跨实现错误静默批跑，也不将资源失败记为模型failure。
            raise
        with progress.open('a') as f:f.write(json.dumps(record)+'\n')
    (out/(a.method+'_queue_done.json')).write_text(json.dumps({'status':'DONE','total_jobs':len(jobs),'elapsed_s':time.time()-started,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')},indent=2))

if __name__=='__main__':main()
