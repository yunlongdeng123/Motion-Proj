"""Pure stdlib follower of one owned training process; one bounded evaluation."""
import argparse,json,time,sys,subprocess,os,datetime
from pathlib import Path
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--evaluation',required=True);a=p.parse_args()
status_path=root/(a.evaluation+'_follow.json')
def write(data):
    data['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=status_path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,status_path)
start=time.time()
while time.time()-start<4*3600:
    path=root/a.train/'status.json'
    if path.exists():
        row=json.loads(path.read_text());write({'state':'waiting_training','training_phase':row.get('phase'),'pid':os.getpid()})
        if row['state'] in ['failed','yielded_to_priority_work']:write({'state':'training_needs_review'});raise SystemExit(2)
        if row['state']=='complete':
            pid=json.loads((root/(a.train+'_launch.json')).read_text())['pid'];cmd=Path('/proc',str(pid),'cmdline')
            if cmd.exists() and cmd.read_bytes():time.sleep(5);continue
            out=root/a.evaluation
            if out.exists():raise SystemExit('Evaluation output already exists; preserve')
            command=[sys.executable,str(root/'implementation/evaluate_sequence.py'),'--output',str(out),'--restore',str(root/a.train/'final.pt')]
            with (root/(a.evaluation+'_console.log')).open('xb') as log:worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            write({'state':'evaluating','pid':os.getpid(),'evaluation_pid':worker.pid,'command':command})
            code=worker.wait();write({'state':'complete' if code==0 else 'evaluation_failed','evaluation_exit_code':code,'pid':os.getpid()});raise SystemExit(code)
    time.sleep(30)
write({'state':'deadline_exceeded'});raise SystemExit(3)
