from pathlib import Path
import base64,json
W=Path(__file__).resolve().parent
files={name:base64.b64encode((W/name).read_bytes()).decode() for name in ['idsplat_frontend_r13.py','idsplat_frontend_r13_protocol.json']}
script='''from pathlib import Path
import base64,json,subprocess,os,time,importlib.util
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'idsplat_frontend_r13';D.mkdir(exist_ok=False)
FILES=PAYLOAD
for name,value in FILES.items():
    p=D/'protocol.json' if name.endswith('_protocol.json') else R/name
    assert not p.exists(),p
    p.write_bytes(base64.b64decode(value))
assert importlib.util.find_spec('sklearn') is not None,'sklearn unavailable, no shared-env modification'
assert (R/'ownership_controls_r12_sam3/USER_STOP_BC_receipt.json').exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (D/'frontend.log').open('xb') as log:
    p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python',str(R/'idsplat_frontend_r13.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
v=dict(pid=p.pid,started_epoch=time.time(),phase='POINTSET_CPU_ONLY',training_updates=0)
(D/'launch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''.replace('PAYLOAD',repr(files))
(W/'deploy_IDSplat_r13_remote.py').write_text(script,encoding='utf8')
