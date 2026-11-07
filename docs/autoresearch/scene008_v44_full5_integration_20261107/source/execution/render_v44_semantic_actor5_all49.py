"""Four-frame native shared-sort integration; zero fitting/training/compositing."""
from pathlib import Path
import sys,os,json,hashlib,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=B/'scene008_v44_full5_integration_20261107_r1';OLD=B/'scene008_v44_rgb_asset_20261107_r1'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
ADAPTER=REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit/v44_integration.py'
EXPECTED='c177c97bea2c73e896fa925d3ecaa8ed490bb5b810e352ed77b2af23c5610c44'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 contract_path=ROOT/'contract/execution_contract_semantic_actor5_all49.json';contract=json.loads(contract_path.read_text())
 assert contract['status']=='SEALED_BEFORE_CANDIDATE_GPU' and contract['evaluation_raws']==list(range(89,138))
 assert sha(ADAPTER)==EXPECTED and (ROOT/'contract/root_baseline4_exact_png_audit.json').exists()
 sys.path[:0]=[str(OLD/'source'),str(REPO)]
 import numpy as np,torch
 from PIL import Image
 from v44_runtime import V44Runtime,tensor_hash
 from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
 assert torch.cuda.is_available();torch.set_num_threads(1)
 start=time.time();base=V44Runtime(OLD/'contract/input_contract.json')
 rt=SemanticActor5V44Integration(base,semantic_sidecar_path=ROOT/'semantic_actor5_calibration49/semantic_actor5_manifest.json',full5_transform_path=ROOT/'contract/full5_transform_matrix.txt')
 with np.load(ROOT/'semantic_actor5_calibration49/semantic_actor5_rows.npz',allow_pickle=False) as z:
  assert np.array_equal(rt.semantic_static_row_ids.detach().cpu().numpy(),z['static_row_ids'])
  assert np.array_equal(rt.semantic_dynamic_row_ids.detach().cpu().numpy(),z['dynamic_row_ids'])
 assert len(rt.semantic_static_row_ids)==232 and len(rt.semantic_dynamic_row_ids)==120722
 outdir=ROOT/'candidate_semantic_a5_49';outdir.mkdir(exist_ok=False);(outdir/'frames').mkdir()
 rows=[]
 for raw in list(range(89,138)):
  data=rt.raw(raw);frame=outdir/f'frames/{raw:03d}';frame.mkdir();states={};added_keep=None;asset_keep=None
  gt=np.asarray(data['obs']['rgb'],dtype=np.float32);gt=gt/255 if gt.max()>1 else gt
  Image.fromarray(np.uint8(np.clip(gt,0,1)*255+.5)).save(frame/'observed.png')
  for state in ['keep','delete']:
   delete=state=='delete'
   with torch.no_grad():
    before=base.scene(raw,delete=delete);scene=rt.scene(raw,delete=delete)
    b=before['rows'];c=scene['rows'];n=len(b['means3D']);assert len(c['means3D'])==n+120954
    shared_before={};shared_after={}
    for key,value in b.items():
     if key=='features':continue
     if value is None:assert c[key] is None;continue
     assert torch.is_tensor(value) and c[key].dtype==value.dtype and c[key][:n].shape==value.shape,key
     assert torch.equal(value,c[key][:n]),key
     shared_before[key]=value;shared_after[key]=c[key][:n]
    before_hash=tensor_hash(shared_before);after_hash=tensor_hash(shared_after);assert before_hash==after_hash
    env_before=tensor_hash({'env':before['env'],'bg':before['bg']});env_after=tensor_hash({'env':scene['env'],'bg':scene['bg']});assert env_before==env_after
    added_hash=tensor_hash({k:v[n:] for k,v in c.items() if k!='features' and v is not None})
    if delete:assert added_hash==added_keep
    else:added_keep=added_hash
    output=rt.render(raw,scene)
    rgb=output['rgb'].detach().cpu().numpy();alpha=output['alpha'].detach().cpu().numpy()[0]
    depth_sum=output['depth_sum'].detach().cpu().numpy()[0];groups=output['group_alpha_T'].detach().cpu().numpy()
    added_rgb=output['added_owner_rgb_T'].detach().cpu().numpy()
    assert rgb.shape==(3,320,480) and groups.shape==(7,320,480) and added_rgb.shape==(3,3,320,480)
    assert all(np.isfinite(x).all() for x in [rgb,alpha,depth_sum,groups,added_rgb])
    assert float(np.abs(groups[5:]).max())==0. and float(np.abs(added_rgb[1:]).max())==0.
    conservation=float(np.abs(groups.sum(0)-alpha).max());assert conservation<2e-5
    meta=output['meta'];asset_hash=meta['native_asset_row_hash']
    if delete:
     assert meta['native_a37_packed_rows']==0 and float(np.abs(groups[1]).max())==0.
     assert asset_hash==asset_keep
    else:asset_keep=asset_hash
    depth=depth_sum/np.maximum(alpha,1e-8)/rt.scale
    np.savez_compressed(frame/(state+'_float.npz'),rgb=rgb,alpha=alpha,depth_m=depth,depth_sum=depth_sum,
        group_alpha_T=groups,added_owner_rgb_T=added_rgb)
    Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(frame/(state+'.png'))
    states[state]=dict(meta=meta,baseline_shared_rows=n,candidate_rows=len(c['means3D']),
        shared_physical_before_hash=before_hash,shared_physical_after_hash=after_hash,
        shared_env_before_hash=env_before,shared_env_after_hash=env_after,
        added_physical_hash=added_hash,asset_row_hash=asset_hash,group_alpha_conservation_maxabs=conservation,
        a37_joint_max=float(np.abs(groups[1]).max()),groups=['asset','a37','road','native_static','a5','a18','a26'],
        added_owner_joint_sums={str(a):float(groups[4+i].astype(np.float64).sum()) for i,a in enumerate([5,18,26])})
    del output,b,c,before,scene,shared_before,shared_after
   rt.assert_unchanged()
  rows.append(dict(raw=raw,states=states));write(frame/'input_guard.json',rows[-1])
  write(outdir/'progress.json',dict(completed=[r['raw'] for r in rows]));print(json.dumps(dict(raw=raw,status='RENDERED_WITH_EXACT_SHARED_ROWS')),flush=True)
  base._data.clear();torch.cuda.empty_cache()
 receipt=dict(status='RENDERED_ALL49_FIXED_SEMANTIC_ACTOR5_DIAGNOSTIC_PENDING_COMPLETE_GATES_AND_ROOT_REVIEW',raws=list(range(89,138)),
    frames=rows,runtime_receipt=rt.receipt(),seconds=time.time()-start,adapter_sha256=sha(ADAPTER),contract_sha256=sha(contract_path),
    new_training_updates=0,new_background_solves=0,pixel_composite=False,quality_accepted=False,human_verdict=None,shutdown=False,reset_card=False)
 write(outdir/'receipt.json',receipt)
if __name__=='__main__':main()
