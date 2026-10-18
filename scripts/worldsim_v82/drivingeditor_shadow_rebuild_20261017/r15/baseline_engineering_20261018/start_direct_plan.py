from pathlib import Path
import subprocess,os,json,datetime
E=Path(__file__).resolve().parent;PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
assert json.loads((E/'direct_CPU_gate.json').read_text())['status']=='PASSED'
jobs=[('direct','run_direct_30k30k.py'),('all_vehicle_coverage','audit_all_vehicle_coverage.py')]
receipts=[]
for name,script in jobs:
 p=E/(name+'_launch.json')
 assert not p.exists(),'registered launch exists; inspect before retry'
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 if name=='all_vehicle_coverage':env['CUDA_VISIBLE_DEVICES']=''
 with (E/(name+'_controller.log')).open('x') as f:
  child=subprocess.Popen([PY,'-B','-u',str(E/script)],cwd=E,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 rec=dict(pid=child.pid,script=str(E/script),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),duplicate_launch_forbidden=True)
 p.write_text(json.dumps(rec,indent=2));receipts.append(rec)
print(json.dumps(receipts))
