"""仅执行观测检查前瞻锁定的两个案例，保留良好与残余结果。"""
import json
import os
import time
from prepare_visible_sources import OUT
from run_natural_readouts import call,DV

def main():
    selection=json.loads((OUT/'observation_selection.json').read_text());assert selection['status']=='complete'
    assert 0<len(selection['model_cases'])<=2 and not (OUT/'readout_queue_result.json').exists()
    rows=[];result={'status':'running','pid':os.getpid(),'cases':rows,'human_verdict':None};began=time.monotonic()
    (OUT/'readout_queue_result.json').write_text(json.dumps(result)+'\n')
    try:
        for row in selection['model_cases']:
            log=row['log_id'];case=OUT/'cases'/log;case.mkdir(parents=True);base=case/'base';read=case/'reconstruction'
            print(json.dumps({'stage':'prepare','log_id':log}),flush=True)
            call('prepare_argoverse.py',case/'prepare.log',args=['--output',base,'--log-id',log,'--start-offset-seconds','.5',
                 '--task-id','WS-V75-VISIBLE-DEV-01','--run-id','20260920-r1'])
            call('prepare_natural.py',case/'prepare_readout.log',args=['--base-dir',base,'--output',read,'--target',row['target'],
                 '--task-id','WS-V75-VISIBLE-DEV-01','--source-protocol',OUT/'protocol.json'])
            print(json.dumps({'stage':'inference','log_id':log}),flush=True)
            call('infer_natural.py',case/'inference.log',python=DV,args=['--run-dir',read])
            call('audit_natural_geometry.py',case/'audit.log',args=['--run-dir',read])
            call('readout_natural.py',case/'readout.log',args=['--run-dir',read,'--calibrated-rays'])
            data=json.loads((read/'readout_ray_control_result.json').read_text())
            record={'log_id':log,'target':row['target'],'readout_dir':str(read),'admitted':data['generation_admitted'],
                    'stop_reasons':data['stop_reasons'],'readouts':data['readouts'],
                    'inference':json.loads((read/'inference_result.json').read_text())}
            rows.append(record);(OUT/'readout_queue_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'stage':'complete','log_id':log,'admitted':record['admitted'],'stop_reasons':record['stop_reasons'],
                              'errors_m':{k:(v['center_error_m'] if v else None) for k,v in record['readouts'].items()}}),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-began;(OUT/'readout_queue_result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
