"""Stage original shared-state targets and fixed masks; fusion is explicitly pending."""
from pathlib import Path
import hashlib,json
import numpy as np
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_road_fusion_bake_20261107_r1';OUT=RUN/'targets'
OLD=CAM/'scene008_road_r0_v44_20261107_r1'
HELD=(89,95,101,105,113,121,125,130,137);BUILD=tuple(x for x in range(89,138) if x not in HELD)
MASKS=OLD/'surface/material_cellfit_strip_v1/supervision_masks.npz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True);original=np.load(MASKS);rows=[];inputs={str(MASKS):sha(MASKS)}
 for raw in range(89,138):
  op=CAM/f'scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz'
  vp=CAM/f'scene008_blue_global_road1k_20261105_r1/inputs/views/{raw:03d}.npz'
  obs=np.load(op);view=np.load(vp)
  rgb=obs['rgb'].astype(np.float32)
  if rgb.max()>1:rgb/=255.
  mask=obs['road'].astype(bool)&~obs['visible'].astype(bool).any(0)&~obs['visible_reviewed'].astype(bool).any(0)&~obs['box_candidate'].astype(bool).any(0)&~view['other_vehicle_guard'].astype(bool)
  if raw in BUILD:assert np.array_equal(mask,original[f'raw{raw:03d}'])
  protect=obs['protect'].astype(bool)&~obs['visible'].astype(bool).any(0)&~mask
  folder=OUT/('projected_BUILD40' if raw in BUILD else 'projected_HELD9');folder.mkdir(exist_ok=True)
  path=folder/f'raw{raw:03d}.npz'
  if path.exists():raise FileExistsError(path)
  np.savez_compressed(path,raw=np.array(raw,np.int16),BUILD40=np.array(raw in BUILD),fusion_ready=np.array(False),
    direct_rgb=rgb,fusion_rgb=rgb.copy(),fit_mask=mask,protect_mask=protect,protect_rgb=rgb,
    fusion_replace_mask=np.zeros_like(mask),atlas_support_mask=np.zeros_like(mask))
  inputs[str(op)]=sha(op);inputs[str(vp)]=sha(vp)
  rows.append(dict(raw=raw,role='BUILD_FIT' if raw in BUILD else 'HELD_PROJECT_EVAL_ONLY',file=str(path.relative_to(OUT)),
    direct_file_sha256=sha(path),fit_pixels=int(mask.sum()),mask_array_sha256=hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()))
 assert sum(r['fit_pixels'] for r in rows if r['raw'] in BUILD)==714546
 receipt=dict(schema='SHARED_WORLD_FUSION_DIRECT_STAGE_V1',status='DIRECT_READY_FUSION_PENDING',BUILD40=list(BUILD),HELD9=list(HELD),
   RGB_encoding='encoded sRGB HWC float32[320,480,3]',state_axis=False,states_share_identical_target_and_masks=True,
   fit_mask='Exact original fixed BUILD40 true-road masks; identical formula for HELD9 evaluation only.',
   protect='Original RGB on obs.protect excluding original visible A6/A37 and fit_mask. Loss usage/weight owned by engineer contract.',
   fusion_placeholder_warning='fusion_rgb currently equals direct_rgb; do not use for fusion formal fit until immutable final target_manifest exists and fusion_ready=true.',
   rows=rows,immutable_input_sha256=inputs,GPU_used=False)
 (OUT/'direct_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
 print(json.dumps({'status':receipt['status'],'BUILD_pixels':714546,'files':49,'manifest':str(OUT/'direct_manifest.json')}))
if __name__=='__main__':main()
