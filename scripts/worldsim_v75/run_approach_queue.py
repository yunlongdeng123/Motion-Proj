"""有限三个误差/控制分支，单卡顺序执行；异常尤其OOM直接停止队列。"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path

OUT=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2')
SOURCE=Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-BASELINE-01/20260920-association-r2')
CONDITIONING=OUT.parent/'20260920-r1/conditioning'


def main():
    global OUT,SOURCE,CONDITIONING
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    parser.add_argument('--source-run',type=Path,default=SOURCE);parser.add_argument('--conditioning-dir',type=Path,default=CONDITIONING)
    parser.add_argument('--task-id',default='WS-V75-APPROACH-CLOSEDLOOP-01')
    args=parser.parse_args();OUT,SOURCE,CONDITIONING=args.run_dir,args.source_run,args.conditioning_dir
    path=OUT/'queue_result.json'; assert not path.exists()
    gt=json.loads((OUT/'gt_clean/result.json').read_text()); assert gt['status']=='complete' and gt['baseline_admitted']
    assert json.loads((OUT/'gt_clean/dense_reference_result.json').read_text())['status']=='passed'
    assert json.loads((OUT/'reconstruction/reference_supplement_result.json').read_text())['generation_admitted']
    result={'status':'running','pid':os.getpid(),'arms':[],'human_verdict':None,'failure_ledger_delta':'none'}; began=time.monotonic()
    def save(): path.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        for arm in ['dvgt_metric','dvgt_lidar_scaled','reference_lidar']:
            print(json.dumps({'stage':'generate','arm':arm}),flush=True)
            with (OUT/f'{arm}.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('run_approach_closed_loop.py')),'--phase','generate','--arm',arm,
                                '--run-dir',str(OUT),'--source-run',str(SOURCE),'--conditioning-dir',str(CONDITIONING),'--task-id',args.task_id],stdout=log,stderr=subprocess.STDOUT,check=True)
            run=json.loads((OUT/arm/'result.json').read_text()); assert run['status']=='complete'
            with (OUT/f'{arm}.assessment.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('assess_following_closed_loop.py')),'--run-dir',str(OUT),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
            dense=json.loads((OUT/arm/'dense_reference_result.json').read_text())
            result['arms'].append({'arm':arm,'result':run,'dense_status':dense['status'],'overlap_frames':dense['overlap_frames']})
            save(); print(json.dumps({'stage':'complete','arm':arm,'overlap_frames':dense['overlap_frames']}),flush=True)
        result['status']='complete'
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        result['wall_s']=time.monotonic()-began; save(); print(json.dumps({k:v for k,v in result.items() if k!='arms'}),flush=True)


if __name__=='__main__': main()
