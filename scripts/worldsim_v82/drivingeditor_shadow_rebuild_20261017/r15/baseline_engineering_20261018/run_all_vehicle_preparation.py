from pathlib import Path
import subprocess,os,json,sys,datetime,traceback
E=Path(__file__).resolve().parent;ROOT=E.parent
PY=ROOT/'shadow_rebuild_20261017_v3/ownership_controls_r12_sam3/runtime_sam3_dinov3/bin/python'
def main():
 assert not (E/'all_vehicle_preparation_started.json').exists()
 (E/'all_vehicle_preparation_started.json').write_text(json.dumps(dict(pid=os.getpid(),user_authorized=True,new_training_budget=30000)))
 env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 with (E/'all_vehicle_native49.log').open('x') as f:subprocess.run([str(PY),'-B','-u',str(E/'infer_all_vehicles_native.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 with (E/'all_vehicle_coverage_fullres.log').open('x') as f:
  subprocess.run([sys.executable,'-B','-u',str(E/'audit_all_vehicle_coverage.py'),'--native-root',str(E/'all_vehicle_native49/native_predictions'),'--output',str(E/'all_vehicle_fullres_coverage'),'--all49-training'],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
 print('FULL49_NATIVE_AND_MEASURED_COVERAGE_READY_FOR_IDENTITY_COMPLETION',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (E/'all_vehicle_preparation_failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_automatic_restart=True)));raise
