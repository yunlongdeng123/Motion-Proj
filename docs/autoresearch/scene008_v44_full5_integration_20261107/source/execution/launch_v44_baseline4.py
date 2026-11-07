from pathlib import Path
import subprocess,sys,json,time,hashlib,os
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=B/'scene008_v44_full5_integration_20261107_r1'
OLD=B/'scene008_v44_rgb_asset_20261107_r1'
ROOT.mkdir(exist_ok=True);(ROOT/'source').mkdir(exist_ok=True);(ROOT/'contract').mkdir(exist_ok=True)
out=ROOT/'baseline4';out.mkdir(exist_ok=False)
script=OLD/'source/evaluate_v44.py';contract=OLD/'contract/input_contract.json'
command=[sys.executable,'-B',str(script),'--worker','v44','--mode','pilot','--contract',str(contract),'--output',str(out)]
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (ROOT/'baseline4.log').open('x') as log:
    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=OLD/'source',env=env,start_new_session=True)
receipt=dict(status='STARTED_FROZEN_ADOPTED_V44_BASELINE4',pid=p.pid,unix=time.time(),command=command,
    raws=[105,113,130,137],states=['keep','delete'],baseline=str(OLD),not_road_r0_candidate=True,
    source_files={str(x):dict(bytes=x.stat().st_size,mtime_ns=x.stat().st_mtime_ns,sha256=hashlib.sha256(x.read_bytes()).hexdigest())
      for x in [script,OLD/'source/v44_runtime.py',contract]},new_training_updates=0,background_rebuild=False,
    source_models_read_only=True,shutdown=False,reset_card=False)
(ROOT/'contract/baseline_launch_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(ROOT/'baseline4.pid').write_text(str(p.pid)+'\n')
print(json.dumps(dict(pid=p.pid,root=str(ROOT),log=str(ROOT/'baseline4.log'),raws=receipt['raws'])))
