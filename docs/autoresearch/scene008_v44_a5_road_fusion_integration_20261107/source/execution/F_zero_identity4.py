"""Real CUDA public bounded-F zero-residual identity; no optimization."""
from pathlib import Path
import hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_v44_a5_road_fusion_integration_20261107_r1';S=B/'scene008_v44_full5_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sys.path[:0]=[str(REPO),str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
import numpy as np
from scripts.worldsim_v82.drivingeditor_dynamic_edit.bounded_road_appearance import write_bounded_appearance_material,BoundedRoadAppearanceV44Integration
api=REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/bounded_road_appearance.py';assert sha(api)=='2f6b5a1cf41a0a82e3d11af00a5faebdd29ffc40750386f22d56c772714c1d81'
contract=R/'contract/bounded_surface_fusion_bake_pilot_v1.json';assert sha(contract)=='37f825b4ba817d3cee18d846e462eb273067f4617006b1b6baf2e337cefae759'
qual=R/'F4_bounded_appearance_qualification/qualification_manifest.json';assert sha(qual)=='c79ad887a9f6e6dbccf77a98575c37a7f601d4e8fee6de452250d9b7ac6c4a96'
material=write_bounded_appearance_material(qual,np.zeros((203932,3),np.float32),R/'F_zero_material')
import torch
from v44_runtime import V44Runtime
from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
from PIL import Image
torch.set_num_threads(1);assert torch.cuda.is_available();outdir=R/'F_zero_identity4';outdir.mkdir(exist_ok=False);(outdir/'frames').mkdir();start=time.time()
base=V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
semantic=SemanticActor5V44Integration(base,semantic_sidecar_path=S/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=R/'contract/full5_transform_matrix.txt')
rt=BoundedRoadAppearanceV44Integration(semantic,qual,material);rows=[];calls=0
for raw in [94,105,128,137]:
 frame=outdir/f'frames/{raw:03d}';frame.mkdir()
 for state in ['keep','delete']:
  scene=rt.scene(raw,delete=state=='delete');out=rt.render(raw,scene);calls+=1;write(outdir/'kernel_progress.json',{'actual_native_forwards':calls,'optimization_updates':0})
  values={'rgb':out['rgb'].detach().cpu().numpy(),'alpha':out['alpha'].detach().cpu().numpy()[0],
   'depth_sum':out['depth_sum'].detach().cpu().numpy()[0],'group_alpha_T':out['group_alpha_T'].detach().cpu().numpy(),
   'added_owner_rgb_T':out['added_owner_rgb_T'].detach().cpu().numpy()}
  values['depth_m']=values['depth_sum']/np.maximum(values['alpha'],1e-8)/base.scale
  errors={}
  with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
   for k,v in values.items():errors[k]=float(np.abs(v-z[k]).max());assert np.array_equal(v,z[k]),(raw,state,k,errors[k])
  np.savez_compressed(frame/(state+'_float.npz'),**values)
  Image.fromarray(np.uint8(np.clip(values['rgb'].transpose(1,2,0),0,1)*255+.5)).save(frame/(state+'.png'))
  rows.append({'raw':raw,'state':state,'all_float_values_exact':True,'max_abs':errors});write(outdir/'progress.json',{'completed':rows,'actual_native_forwards':calls});print(json.dumps(rows[-1]),flush=True);rt.assert_unchanged();del scene,out
 base._data.clear();torch.cuda.empty_cache()
rt.assert_unchanged(full=True)
write(outdir/'receipt.json',{'status':'PASS_PUBLIC_BOUNDED_F_ZERO_DELTA_ALL4_KD_EXACT_BASELINE','rows':rows,'actual_native_forwards':calls,'optimizer_steps':0,
 'original_model_updates':0,'new_residual_updates':0,'api_sha256':sha(api),'fit_contract_sha256':sha(contract),'material':str(material),'runtime_receipt':rt.receipt(),'seconds':time.time()-start})
print(json.dumps({'status':'PASS_ZERO_IDENTITY4','actual_native_forwards':calls,'optimizer_steps':0}),flush=True)
