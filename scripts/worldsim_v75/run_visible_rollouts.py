"""最多两个前瞻锁定案例的四组生成；任一进程失败立即停止。"""
import json
import os
import time
from prepare_visible_sources import OUT
from run_natural_readouts import call

def main():
    read=json.loads((OUT/'readout_queue_result.json').read_text());assert read['status']=='complete'
    path=OUT/'rollout_queue_result.json';assert not path.exists()
    result={'status':'running','pid':os.getpid(),'completed':[],'skipped':[],'human_verdict':None};began=time.monotonic()
    path.write_text(json.dumps(result)+'\n')
    try:
        for row in read['cases']:
            log=row['log_id'];case=OUT/'cases'/log
            if not row['admitted']:
                result['skipped'].append({'log_id':log,'reasons':row['stop_reasons']});continue
            print(json.dumps({'stage':'four_arm_generation','log_id':log}),flush=True)
            call('run_natural_rollouts.py',case/'rollouts.log',args=['--visible-case',case])
            call('review_natural_rollouts.py',case/'review.log',args=['--run-dir',case/'rollouts'])
            result['completed'].append(log);path.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'stage':'review_complete','log_id':log}),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-began;path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
