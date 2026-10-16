from pathlib import Path
import base64
W=Path(__file__).resolve().parent;payload=base64.b64encode((W/'dinov3_registration_r13.py').read_bytes()).decode()
s='''from pathlib import Path
import base64,subprocess,json,os,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3');D=R/'idsplat_frontend_r13';S=R/'ownership_controls_r12_sam3'
assert (D/'frontend_complete.json').exists()
assert not (D/'dinov3_launch.json').exists(),'do not duplicate GPU validation'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'other GPU task first'
p=R/'dinov3_registration_r13.py';assert not p.exists();p.write_bytes(base64.b64decode(PAYLOAD))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
with (D/'dinov3.log').open('xb') as log:
    worker=subprocess.Popen(['flock','-n','/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock',str(S/'runtime_sam3_dinov3/bin/python'),str(p)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(pid=worker.pid,started_epoch=time.time(),purpose='real DINOv3 and official100k RANSAC independent validation only',scene_training_updates=0)
(D/'dinov3_launch.json').write_text(json.dumps(value,indent=2));print(json.dumps(value))
'''.replace('PAYLOAD',repr(payload))
(W/'deploy_DINO_r13_remote.py').write_text(s,encoding='utf8')
