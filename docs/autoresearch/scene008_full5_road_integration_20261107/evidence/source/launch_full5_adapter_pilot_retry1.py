from pathlib import Path
import json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full5_road_integration_20261107_r1');env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
with (R/'adapter_C4_retry1_queue.log').open('x') as log:p=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B',str(R/'source/run_full5_adapter_pilot_retry1.py')],stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
failure={'status':'PRESERVED_FIRST_NO_C_FORWARD_METRIC_UNIT_POSTPROCESS_ASSERTION','actual_native_joint_calls':1,'actual_raster_kernel_forwards':1,'backwards':0,'optimizer_updates':0,'cause':'CPU NumPy versus CUDA division rounding in derived metres; raw depth_sum and alpha were exact','correction':'Same baseline NumPy postprocess for exported depth_m; API raw CUDA depth_m retained separately in depth_m_gpu','API_geometry_or_road_asset_changed':False}
(R/'contract/adapter_parity4_first_forward_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
record={'pid':p.pid,'unix':time.time(),'automatic_training':False,'automatic49':False};(R/'contract/adapter_C4_retry1_queue_launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
