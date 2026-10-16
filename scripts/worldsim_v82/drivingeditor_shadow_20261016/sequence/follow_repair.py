import time,json,sys,subprocess,os,datetime
from pathlib import Path
NEW=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)
start=time.time();statusfile=NEW/'repair_follow_status.json'
while time.time()-start<4*3600:
    path=NEW/'repair_union_v1/status.json'
    if path.exists():
        row=json.loads(path.read_text())
        write(statusfile,{'state':'waiting_repair','phase':row.get('phase'),'updated_utc':now(),'pid':os.getpid()})
        if row['state'] in ['failed','yielded_to_priority_work']:raise SystemExit(2)
        if row['state']=='complete':
            pid=json.loads((NEW/'repair_union_v1_launch.json').read_text())['pid']
            if Path('/proc',str(pid),'cmdline').exists() and Path('/proc',str(pid),'cmdline').read_bytes():time.sleep(5);continue
            out=NEW/'final49_repair_v1'
            if out.exists():raise SystemExit('Preserve existing evaluation')
            cmd=[sys.executable,str(NEW/'implementation/evaluate_sequence.py'),'--output',str(out),'--restore',str(NEW/'repair_union_v1/final.pt')]
            with (NEW/'final49_repair_v1_console.log').open('xb') as f:
                p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            write(statusfile,{'state':'evaluating_repair','pid':os.getpid(),'evaluation_pid':p.pid,'command':cmd,'updated_utc':now()})
            code=p.wait();write(statusfile,{'state':'complete' if code==0 else 'evaluation_failed','evaluation_exit_code':code,'updated_utc':now(),'pid':os.getpid()});raise SystemExit(code)
    time.sleep(30)
raise SystemExit('Repair monitoring deadline')
