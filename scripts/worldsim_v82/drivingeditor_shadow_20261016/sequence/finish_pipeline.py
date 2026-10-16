"""Bounded ordinary process monitor; never invokes an agent or trains extra steps."""
import os,sys,time,json,subprocess,datetime
from pathlib import Path
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
receipt=root/'finish_pipeline_status.json'
start=time.time()
def write(data):
    data['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=receipt.with_suffix('.partial');p.write_text(json.dumps(data,indent=2));os.replace(p,receipt)
while time.time()-start<4*3600:
    path=root/'joint_v1/status.json'
    if path.exists():
        status=json.loads(path.read_text());state=status['state']
        write({'state':'waiting_training','training_state':state,'training_phase':status.get('phase'),'pid':os.getpid()})
        if state in ['failed','yielded_to_priority_work']:
            write({'state':'needs_review','training_state':state,'pid':os.getpid()});raise SystemExit(2)
        if state=='complete':
            pid=json.loads((root/'joint_v1_launch.json').read_text())['pid']
            if Path('/proc',str(pid)).exists():time.sleep(5);continue
            out=root/'final49_v1'
            if out.exists():raise SystemExit('Preserve existing final49 evaluation')
            cmd=[sys.executable,str(root/'implementation/evaluate_sequence.py'),'--output',str(out),'--restore',str(root/'joint_v1/final.pt')]
            with (root/'final49_v1_console.log').open('xb') as f:
                worker=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            write({'state':'evaluating','pid':os.getpid(),'evaluation_pid':worker.pid,'command':cmd})
            code=worker.wait();write({'state':'complete' if code==0 else 'evaluation_failed','evaluation_exit_code':code,'pid':os.getpid()});raise SystemExit(code)
    time.sleep(30)
write({'state':'deadline_exceeded','pid':os.getpid()});raise SystemExit(3)
