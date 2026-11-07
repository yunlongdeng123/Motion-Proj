from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full5_road_integration_20261107_r1');PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
protocol={'status':'SEALED_BEFORE_FULL5_ADAPTER_AND_C_PILOT_GPU','raw_ids':[94,105,128,137],'states':['keep','delete'],'single_model_source':'Full5@10000 only',
 'adapter_source_sha256':sha(R/'source/full5_road_scene.py'),'worker_source_sha256':sha(R/'source/render_full5_adapter4.py'),
 'parity_RGB_max_abs_tolerance':1e-6,'parity_alpha_depth_original6groups':'exact','C_geometry':'all40392 original C rows, no row/color fitting',
 'C_native_full5_prefix':'all prepared fields including masks unchanged from noC packet','quality_gate':'Structure/deletion/depth and road acceptance require actual root image review; no automatic training or49',
 'new_native_kernel_forwards_planned':16,'optimizer_updates':0,'shutdown':False}
write(R/'contract/adapter_and_C4_protocol_v1.json',protocol)
for variant,stem in [('parity','adapter_parity4'),('C','C4')]:
 cmd=[PY,'-B',str(R/'source/render_full5_adapter4.py'),'--variant',variant]
 with (R/(stem+'.log')).open('x') as log:p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=R,env=env,start_new_session=True,stdin=subprocess.DEVNULL)
 write(R/'contract'/(stem+'_launch.json'),{'pid':p.pid,'command':cmd,'unix':time.time(),'native_kernel_forwards_planned':8,'optimizer_updates':0})
 print(json.dumps({'variant':variant,'pid':p.pid}),flush=True)
 code=p.wait()
 if code:write(R/'adapter_C4_queue_status.json',{'status':'FAILED_PRESERVED','variant':variant,'code':code,'automatic_training':False});raise SystemExit(code)
write(R/'adapter_C4_queue_status.json',{'status':'COMPLETE_PARITY_AND_C4_PENDING_STRUCTURAL_GATE','actual_native_kernel_forwards':16,'optimizer_updates':0,'automatic_training':False,'automatic49':False,'shutdown':False})
