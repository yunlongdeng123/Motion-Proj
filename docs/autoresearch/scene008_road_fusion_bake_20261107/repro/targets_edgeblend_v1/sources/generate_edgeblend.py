"""唯一固定10px边缘融合控制；复制原目标，不改原target或Poisson。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.ndimage import distance_transform_edt
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_fusion_bake_20261107_r1')
SRC=RUN/'targets';OUT=RUN/'targets_edgeblend_v1';EXPECTED='d08232bb8df0ca6cf4a2f986c36074b9a588e54d346b97188aa0c716b2a83bea'
WIDTH=10.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def stats(x):
 x=np.asarray(x,dtype=float)
 return dict(count=int(x.size),sum=float(x.sum()),mean=float(x.mean()) if x.size else None,p50=float(np.median(x)) if x.size else None,p95=float(np.quantile(x,.95)) if x.size else None,max=float(x.max()) if x.size else None)
def main():
 start=time.time();assert sha(SRC/'target_manifest.json')==EXPECTED
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'target_manifest.json').exists() or (OUT/'edgeblend_contract.json').exists():raise FileExistsError('preserve first fixed-width control')
 parent=json.loads((SRC/'target_manifest.json').read_text())
 contract=dict(schema='FIXED10PX_VALIDITY_EDGE_FUSION_CONTRACT_V1',task_run=RUN.name,
   parent_target_manifest_sha256=EXPECTED,parent_atlas_contract_sha256=sha(SRC/'atlas_contract.json'),
   width_pixels=WIDTH,mask='Exactly old fusion_replace_mask M, including internal holes and all boundary classes. No dilation/filling or source changes.',
   formula='C=O+clip(Euclidean distance_inside(M)/10,0,1)*(B-O)',
   roles={'edgeblend_rgb':'C actual new complete-joint-image RGB target for the single fallback',
          'fusion_rgb':'Original locked B Poisson projection copied unchanged, never aliased to C',
          'direct_rgb':'Original locked A sourcechart projection copied unchanged',
          'observation_rgb':'Original actual rawRGB O; legal current-view anchor, not hidden roadGT',
          'edgeblend_weight':'Fixed per-view validity-distance weight, no learned/free image colour'},
   selected_target_key='edgeblend_rgb',RGB_encoding='encoded sRGB float32 HWC[320,480,3]',
   source='Same190 nonHELD FRONT source witnesses, same fixed Poisson atlas and measured geometry, same old fit/protect/M masks.',
   state_axis=False,KEEP_DELETE_same_target=True,HELD9_policy='Projection/evaluation only; no atlas or border fitting',
   shared_scope='Atlas B remains one shared worldtexture. C is a deterministic view-conditioned observation/validity blend, not a pure world-only colour target.',
   required_description='Strong realRGB anchoring / source-prior dilution. Report actual mean/max weight and no full-atlas-weight pixels; do not call algebraic mask-edge reduction native success.',
   comparison='Only new actual fallback C vs already executed hardB. Width10 fixed once, no grid, no legacy extra trial. Joint3D fit400pairedsteps remains engineer/root owned.',
   frozen='Original targets/manifest, source/Poisson parameters, geometry/alpha/camera and original fields remain unchanged.',
   extra_GPU_forwards=0,source_code_sha256=sha(__file__),failure_ledger_refs=['V82-F13'],human_verdict=None)
 write(OUT/'edgeblend_contract.json',contract)
 rows=[];max_replay=0.
 for row in parent['rows']:
  original=SRC/row['file'];assert sha(original)==row['sha256']
  with np.load(original,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
  M=arrays['fusion_replace_mask'];O=arrays['observation_rgb'];B=arrays['fusion_rgb']
  weight=np.minimum(distance_transform_edt(M)/WIDTH,1).astype(np.float32)
  C=(O+weight[:,:,None]*(B-O)).astype(np.float32)
  assert np.isfinite(C).all() and C.min()>=0 and C.max()<=1
  assert np.array_equal(C[~M],O[~M]) and not (M&~arrays['fit_mask']).any()
  arrays.update(edgeblend_rgb=C,edgeblend_weight=weight,edgeblend_ready=np.array(True))
  path=OUT/row['file'];path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arrays)
  with np.load(path,allow_pickle=False) as check:
   for key in arrays:
    assert np.array_equal(check[key],arrays[key],equal_nan=True),key
   for key in ['direct_rgb','fusion_rgb','observation_rgb','fit_mask','protect_mask','fusion_replace_mask']:
    with np.load(original,allow_pickle=False) as old:assert np.array_equal(check[key],old[key],equal_nan=True),key
  meta={k:v for k,v in row.items() if k!='sha256'}
  meta.update(sha256=sha(path),parent_file_sha256=row['sha256'],selected_target_key='edgeblend_rgb',
     edgeblend_weight_on_M=stats(weight[M]),full_atlas_weight_pixels=int((weight[M]>=1).sum()),
     edgeblend_vs_observation_on_M=stats(abs(C[M]-O[M]).mean(1)),
     edgeblend_vs_parentB_on_M=stats(abs(C[M]-B[M]).mean(1)),
     outside_M_equal_observation=True,original_A_B_O_masks_unchanged=True)
  if row['raw']==128:
   roi=np.zeros(M.shape,bool);roi[188:234,165:208]=True
   meta['primary_blue_bbox_weight']=stats(weight[M&roi])
  rows.append(meta);print(json.dumps(dict(raw=row['raw'],seconds=time.time()-start)),flush=True)
 grouped=[]
 for role in ['BUILD_FIT','HELD_PROJECT_EVAL_ONLY']:
  selected=[r for r in rows if r['role']==role];n=sum(r['edgeblend_weight_on_M']['count'] for r in selected);s=sum(r['edgeblend_weight_on_M']['sum'] for r in selected)
  grouped.append(dict(role=role,replace_pixels=n,mean_weight=s/n if n else None,max_weight=max(r['edgeblend_weight_on_M']['max'] or 0 for r in selected),
     full_atlas_weight_pixels=sum(r['full_atlas_weight_pixels'] for r in selected),
     edgeblend_vs_trueRGB_on_M=dict(count_pixels=n,sum_pixel_mean_absRGB=sum(r['edgeblend_vs_observation_on_M']['sum'] for r in selected),
       mean=sum(r['edgeblend_vs_observation_on_M']['sum'] for r in selected)/n if n else None)))
 manifest=dict(schema='IMMUTABLE_FIXED10PX_EDGE_BLEND_TARGETS_V1',status='FINAL_SHARED_TARGETS_READY',fusion_ready=True,edgeblend_ready=True,
   selected_target_key='edgeblend_rgb',BUILD40=parent['BUILD40'],HELD9=parent['HELD9'],rows=rows,
   parent_target_manifest_sha256=EXPECTED,edgeblend_contract_sha256=sha(OUT/'edgeblend_contract.json'),
   fit_pixels_unique_BUILD40=parent['fit_pixels_unique_BUILD40'],replace_pixels_BUILD40=parent['replace_pixels_BUILD40'],
   common_data_masks_exact_parent=True,state_axis=False,KEEP_DELETE_same_target_mask=True,
   group_weight_summary=grouped,interpretation='Strong realRGB anchoring/dilution; full image target is view-conditioned, underlying atlas stays world-shared.',
   original_targets_modified=False,new_GPU_forwards=0,seconds=time.time()-start,failure_ledger_refs=['V82-F13'],human_verdict=None)
 assert sha(SRC/'target_manifest.json')==EXPECTED
 write(OUT/'target_manifest.json',manifest)
 write(OUT/'target_preflight_0F.json',dict(status='PASS_FIXED10PX_TARGET_BUILD_0F',new_GPU_forwards=0,files=49,
   final_manifest_sha256=sha(OUT/'target_manifest.json'),source_manifest_unchanged_sha256=EXPECTED,
   original_A_B_O_masks_exact=True,outside_equal_trueRGB=True,weights=grouped,
   primary_raw128=next(r['primary_blue_bbox_weight'] for r in rows if r['raw']==128),targets_are_GT=False,whole_scene_adopted=False))
 print(json.dumps(dict(status='FINAL_EDGEBLEND_READY',manifest_sha256=sha(OUT/'target_manifest.json'),group_weights=grouped)),flush=True)
if __name__=='__main__':main()
