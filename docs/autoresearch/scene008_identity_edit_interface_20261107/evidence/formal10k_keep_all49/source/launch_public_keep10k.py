from pathlib import Path
import subprocess,sys,json,time,hashlib,os
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
assert not (ROOT/'cli_keep10k').exists()
assert json.loads((ROOT/'formal10k/returned.json').read_text())['iteration']==10000
command=[sys.executable,'-B','-m','scripts.worldsim_v82.drivingeditor_dynamic_edit.query',
    '--model',str(ROOT/'formal10k'),'--iteration','10000','--output',str(ROOT/'cli_keep10k')]
env=os.environ.copy()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[key]='1'
with (ROOT/'cli_keep10k.log').open('x') as log:
    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=REPO,env=env,start_new_session=True)
receipt=dict(status='STARTED_PUBLIC_SELF_CONTAINED_NATIVE_KEEP10K',pid=p.pid,unix=time.time(),command=command,
    implicit_default_delete_actor_ids=[],raw_ids=[97,105,109],requires_GT_PLY_masks=False,optimization_updates=0,
    query_module_sha256=hashlib.sha256((REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/query.py').read_bytes()).hexdigest(),
    shutdown=False,reset_card=False)
(ROOT/'cli_keep10k_launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
