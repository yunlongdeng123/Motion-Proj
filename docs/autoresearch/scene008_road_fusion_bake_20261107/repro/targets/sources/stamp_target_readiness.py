"""仅schema接线：保留首manifest、核49数组，显式顶层ready；不改目标数值。"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1/targets')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=B/'target_manifest.json';d=json.loads(p.read_text());backup=B/'target_manifest.schema_v0.json'
 if backup.exists():raise FileExistsError('preserve first readiness stamp')
 assert d['status']=='FINAL_SHARED_TARGETS_READY' and len(d['rows'])==49
 probes=[]
 for row in d['rows']:
  file=B/row['file'];assert sha(file)==row['sha256'];z=np.load(file)
  assert bool(z['fusion_ready']) and z['direct_rgb'].shape==z['fusion_rgb'].shape==z['observation_rgb'].shape==(320,480,3)
  assert np.array_equal(z['direct_rgb'][~z['fusion_replace_mask']],z['observation_rgb'][~z['fusion_replace_mask']])
  assert np.array_equal(z['fusion_rgb'][~z['fusion_replace_mask']],z['observation_rgb'][~z['fusion_replace_mask']])
  assert not (z['fusion_replace_mask']&~z['fit_mask']).any()
  assert all(np.isfinite(z[k]).all() and z[k].min()>=0 and z[k].max()<=1 for k in ['direct_rgb','fusion_rgb','observation_rgb'])
  if row['raw']==128:
   roi=np.zeros((320,480),bool);roi[188:234,165:208]=True
   probes.append(dict(raw=128,roi_fit_pixels=int((roi&z['fit_mask']).sum()),roi_replace_pixels=int((roi&z['fusion_replace_mask']).sum()),
     fusion_vs_direct_MAE_ROI=float(abs(z['fusion_rgb'][roi]-z['direct_rgb'][roi]).mean())))
 shutil.copy2(p,backup);oldsha=sha(backup)
 d.update(fusion_ready=True,readiness_schema='TOP_LEVEL_READY_V1',readiness_adapter_source_sha256=sha(__file__),
   first_manifest_schema_v0_sha256=oldsha,target_arrays_modified_by_adapter=False)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 r=dict(status='PASS_FINAL_TARGET_READINESS_SCHEMA_ADAPTER',files_verified=49,target_arrays_modified=False,
   first_manifest_sha256=oldsha,final_manifest_sha256=sha(p),raw128_fixed_roi_probe=probes)
 (B/'schema_adapter_receipt.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r))
if __name__=='__main__':main()
