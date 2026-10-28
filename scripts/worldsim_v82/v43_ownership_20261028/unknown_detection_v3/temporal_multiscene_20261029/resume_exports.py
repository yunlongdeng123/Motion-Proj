from pathlib import Path
import json,datetime,shutil,subprocess,sys,os
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
CTX=['14705303724557273004_3105_000_3125_000','4295449061847708198_3769_000_3789_000','473735159277431842_630_095_650_095']
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=ROOT/'authorization.json';s=json.loads(p.read_text());b=ROOT/'authorization_history'/(stamp.replace(':','')+'_multiscene.json');b.parent.mkdir(exist_ok=True);shutil.copy2(p,b)
s.update(latest_user_scope='User accepted actual bounded R2 visual baseline; requests temporal diagnostics and scene2/3 equivalent V43 edits; GPU enabled explicitly, parallel authorized',
    held_contexts=[],active_development_contexts=CTX,detection_contexts=CTX,allow_target_detection=True,allow_temporal_multiscene=True,
    temporal_contexts=CTX,allow_v43_training=False,old_queue_resumed=False,old_automation_resumed=False,gpu_execution_hold=False,user_research_hold=False,updated_utc=stamp)
p.write_text(json.dumps(s,indent=2));out=ROOT/'temporal_multiscene_20261029';out.mkdir(exist_ok=True)
jobs=[('export_actor_delivery.py',['--name','third_actor_delivery10k_v2','--context',CTX[2],'--steps','10000']),
      ('export_actor_motion.py',['--name','second_motion_delivery_v2','--context',CTX[1]]),
      ('export_actor_motion.py',['--name','third_motion_delivery_v2','--context',CTX[2]])]
for script,args in jobs:
    name=args[args.index('--name')+1]
    if (ROOT/(name+'_receipt.json')).exists():continue
    with (out/(name+'.log')).open('w') as f:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        subprocess.run([sys.executable,str(ROOT/'code'/script)]+args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('EXPORTED',name,flush=True)
(out/'export_status.json').write_text(json.dumps(dict(phase='complete',training_updates=0,reused_completed_before_hold=True,jobs=[args[1] for _,args in jobs]),indent=2))
