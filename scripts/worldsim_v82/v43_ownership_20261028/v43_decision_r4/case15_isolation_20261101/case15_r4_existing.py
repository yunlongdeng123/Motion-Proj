import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pathlib,time
import numpy as np
from PIL import Image
H=pathlib.Path('/root/autodl-tmp/codex_tmp/v43_case15_isolation_20261101_r1');O=H/'r4_existing';O.mkdir(exist_ok=True)
GPU=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/v43_decision_r4_gpu_20261031/run_r1/real_first_scene_native30k/enumerate_r1')
P=pathlib.Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027/motion_scene_inputs_v1/14705303724557273004_3105_000_3125_000')
report={'case':'waymo-train-212/a0','mode':'CPU read-only reuse of existing frozen native GPU arrays; no new native rendering','rows':[],'source_GPU':str(GPU),'limits':['body ROI is associated input mask, not independent owner or edited-background GT','mean RGB/fill amplitude is a component diagnostic, not semantic car ghost proof','three anchors only; fixed H0 enumerated but no trained decisioner or calibrated admission threshold']}
def array(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def rgb(x):return np.rint(np.clip(np.moveaxis(x,0,-1),0,1)*255).astype(np.uint8)
for fid in ['9001077','9001101','9001125']:
 orig=array(GPU/'native_original_none'/(fid+'.npz'));shape=orig['final'].shape[-2:]
 mask=np.asarray(Image.open(P/'dynamic_actor_assets/actor_0/body_masks'/(fid+'.png')).resize((shape[1],shape[0]),Image.Resampling.NEAREST))>0
 assert mask.any();Image.fromarray((mask*255).astype(np.uint8)).save(O/(fid+'_body_roi.png'))
 for hid in ['h000','h007']:
  for op in ['none','delete','shift','rotate']:
   p=GPU/'routes'/hid/op/(fid+'.npz');a=array(p);cg=a['C_G'];fill=a['fill'];alpha=a['alpha'];final=a['final']
   rec={'frame_id':fid,'raw':int(fid[-3:]),'route':hid,'operation':op,'ROI_pixels':int(mask.sum()),'source':str(p),'finite_all_fields':all(bool(np.isfinite(v).all()) for v in a.values()),'closure_final_CG_plus_fill_max_abs':float(np.max(abs(final-cg-fill))),'closure_fill_one_minus_alpha_E_max_abs':float(np.max(abs(fill-(1-alpha)*a['E']))),'environment_E_vs_original_max_abs':float(np.max(abs(a['E']-orig['E']))),'body_alpha_mean':float(alpha[:,mask].mean()),'body_CG_abs_mean':float(abs(cg[:,mask]).mean()),'body_environment_fill_abs_mean':float(abs(fill[:,mask]).mean()),'body_vs_original_final_MAE':float(abs(final[:,mask]-orig['final'][:,mask]).mean()),'body_environment_fill_change_MAE':float(abs(fill[:,mask]-orig['fill'][:,mask]).mean()),'body_CG_change_MAE':float(abs(cg[:,mask]-orig['C_G'][:,mask]).mean())}
   if hid=='h000' and op=='none':rec['allkeep_native_none_parity_max_abs']=float(np.max(abs(final-orig['final'])))
   report['rows'].append(rec)
   if op in ['none','delete']:
    for k in ['final','C_G','fill']:Image.fromarray(rgb(a[k])).save(O/(fid+'_'+hid+'_'+op+'_'+k+'.png'))
(O/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({'rows':len(report['rows']),'closure_max':max(x['closure_final_CG_plus_fill_max_abs'] for x in report['rows']),'allkeep_parity':[x.get('allkeep_native_none_parity_max_abs') for x in report['rows'] if 'allkeep_native_none_parity_max_abs' in x]},ensure_ascii=False),flush=True)
