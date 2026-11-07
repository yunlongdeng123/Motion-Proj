"""CPU-only target readiness. This does not authorize or run colour fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');ROOT=B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
TARGET=B/'scene008_road_fusion_bake_20261107_r1/targets_edgeblend_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for raw in [94,105,128,137]:
 origin='projected_BUILD40' if raw in [94,128] else 'projected_HELD9'
 p=TARGET/origin/f'raw{raw:03d}.npz'
 with np.load(p,allow_pickle=False) as z:
  target=z['edgeblend_rgb'];fit=z['fit_mask'];protect=z['protect_mask']
  assert target.shape==(320,480,3) and fit.shape==protect.shape==(320,480)
  assert np.isfinite(target).all() and set(np.unique(fit).tolist())<=set([False,True])
  assert set(np.unique(protect).tolist())<=set([False,True]) and bool(fit.any())
  rows.append({'raw':raw,'target_file':str(p),'target_sha256':sha(p),'source_original_role':origin,
   'present_role':'CPU preparation for a possible explicitly registered four-view BUILD bake; not yet used for fitting',
   'target_dtype':str(target.dtype),'target_min':float(target.min()),'target_max':float(target.max()),
   'legal_fit_pixels':int(fit.sum()),'protect_pixels':int(protect.sum()),'fit_and_protect_pixels':int((fit.astype(bool)&protect.astype(bool)).sum()),
   'shared_target_for_keep_delete':True,'query_original_RGB_used_by_future_runtime':False})
out=ROOT/'review/E4_existing_C_target_readiness_cpu.json';assert not out.exists()
a={'status':'PASS_CPU_TARGET_SHAPE_FINITE_AND_FIXED_MASK_READINESS_NO_FIT','rows':rows,'target_construction_uses_source_observations':True,
 'original_HELD105_137_role_preserved_in_provenance':True,'fitting_would_require_new_explicit_BUILD_role_contract':True,
 'no_safe_row_or_target_pixel_selection_changed':True,'planned_fit_not_executed':True,'optimizer_steps':0,'new_GPU_forwards':0,'model_updates':0,'colour_updates':0}
out.write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a))
