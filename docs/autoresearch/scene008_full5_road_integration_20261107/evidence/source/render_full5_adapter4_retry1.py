"""One Full5 native packet, first parity then optional complete-C shared sort."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_full5_road_integration_20261107_r1';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');RAWS=[94,105,128,137]
EXPECTED='bdf181f614ec7ea19dddabccf900f97dd091323b914b5f1367f30d53ba5ed17d'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['parity','C'],required=True);a=p.parse_args();variant=a.variant
 import numpy as np,torch
 from PIL import Image
 sys.path[:0]=[str(R/'source'),str(REPO)]
 from full5_road_scene import Full5RoadScene,C_MANIFEST
 source=R/'source/full5_road_scene.py';assert sha(source)==EXPECTED
 if variant=='C':assert json.loads((R/'adapter_parity4_retry1/receipt.json').read_text())['status']=='PASS_FULL5_SINGLE_JOINT_NATIVE_PARITY4'
 outdir=R/('adapter_parity4_retry1' if variant=='parity' else 'C4_retry1');outdir.mkdir(exist_ok=False);(outdir/'frames').mkdir();start=time.time()
 torch.set_num_threads(1);rt=Full5RoadScene(road_manifest=None if variant=='parity' else C_MANIFEST,transform_path=R/'contract/full5_transform_matrix.txt')
 assert rt.dynamic_actor_ids==[5,6,18,26,37]
 def packet_hash(rows,n):
  h=hashlib.sha256()
  for key in sorted(rows):
   v=rows[key];h.update(key.encode()+b'\0')
   if v is None:h.update(b'NONE');continue
   v=v[:n].detach().cpu().contiguous();h.update(str(v.dtype).encode()+str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
  return h.hexdigest()
 calls=0;frames=[];allheaders=[]
 for raw in RAWS:
  folder=outdir/f'frames/{raw:03d}';folder.mkdir();states={};header=rt.headers[raw]
  K=[[header['fx'],0.,header['cx']],[0.,header['fy'],header['cy']],[0.,0.,1.]]
  allheaders.append({'raw':raw,'K':K,'model_to_camera':header['W'].tolist(),'model_time':header['timestamp'],'group_actor_ids':[-1,5,6,18,26,37,-2],'world_to_model_scale':rt.scale})
  for state in ['keep','delete']:
   with torch.no_grad():
    scene=rt.scene(raw,delete=state=='delete');n=scene['meta']['road_offset'];assert n==1469977+1007477
    prefix=packet_hash(scene['rows'],n)
    if variant=='C':
     prior=json.loads((R/f'adapter_parity4_retry1/frames/{raw:03d}/input_guard.json').read_text())['states'][state]
     assert prefix==prior['native_full5_packet_sha256']
     assert scene['meta']['road_rows']==40392 and len(scene['rows']['mask'])==n+40392
    else:assert scene['meta']['road_rows']==0 and len(scene['rows']['mask'])==n
    out=rt.render(raw,scene);calls+=1;write(outdir/'kernel_progress.json',{'actual_native_joint_calls':calls,'actual_raster_kernel_forwards':calls,'backwards':0,'optimizer_updates':0})
    rgb=out['rgb'].detach().cpu().numpy();alpha=out['alpha'].detach().cpu().numpy()[0];depthsum=out['depth_sum'].detach().cpu().numpy()[0];depth=out['depth_m'].detach().cpu().numpy()[0]
    depth_gpu=depth.copy();depth=depthsum/np.maximum(alpha,1e-8)/rt.scale
    groups=out['group_alpha_T'].detach().cpu().numpy();newrgb=out['newroad_rgb_T'].detach().cpu().numpy();assert groups.shape==(7,320,480)
    assert np.isfinite(rgb).all() and np.isfinite(depth).all() and np.isfinite(groups).all()
    errors={}
    if variant=='parity':
     assert float(np.abs(groups[6]).max())==0
     with np.load(R/f'baseline4/frames/{raw:03d}/{state}_float.npz',allow_pickle=False) as z:
      for name,v in [('rgb',rgb),('alpha',alpha),('depth_sum',depthsum),('depth_m',depth),('group_alpha_T',groups[:6])]:
       errors[name]=float(np.abs(v-z[name]).max())
       if name=='rgb':assert errors[name]<=1e-6,(raw,state,name,errors[name])
       else:assert np.array_equal(v,z[name]),(raw,state,name,errors[name])
    if state=='delete':assert float(np.abs(groups[5]).max())==0
    np.savez_compressed(folder/(state+'_float.npz'),rgb=rgb,alpha=alpha,depth_sum=depthsum,depth_m=depth,depth_m_gpu=depth_gpu,group_alpha_T=groups,newroad_rgb_T=newrgb,
     group_actor_ids=np.asarray([-1,5,6,18,26,37,-2],np.int64),raw=np.int64(raw),K=np.asarray(K),model_to_camera=header['W'],model_time=np.float64(header['timestamp']))
    Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(folder/(state+'.png'))
    states[state]={'native_full5_packet_sha256':prefix,'road_rows':scene['meta']['road_rows'],'native_parity_max_abs':errors,'metric_depth_formula':'depth_sum_numpy / maximum(alpha_numpy,1e-8) / world_to_model_scale','GPU_to_NumPy_metric_depth_max_abs':float(np.abs(depth_gpu-depth).max()),'group_conservation_max_abs':out['group_conservation_maxabs'],'meta':scene['meta']}
    del out,scene
  frames.append({'raw':raw,'states':states});write(folder/'input_guard.json',frames[-1]);write(outdir/'progress.json',{'completed_raw_ids':[r['raw'] for r in frames],'actual_raster_kernel_forwards':calls});print(json.dumps({'raw':raw,'variant':variant,'status':'RENDERED_BOTH','actual_kernel_forwards':calls}),flush=True)
  rt._raw.clear();torch.cuda.empty_cache()
 rt.assert_unchanged(full=True);write(outdir/'frame_headers.json',{'frames':allheaders})
 result={'status':'PASS_FULL5_SINGLE_JOINT_NATIVE_PARITY4' if variant=='parity' else 'RENDERED_FULL5_ONLY_COMPLETE_C4_PENDING_QUALITY','variant':variant,'model':str(rt.model),'iteration':10000,'raw_ids':RAWS,'frames':frames,
  'group_actor_ids':[-1,5,6,18,26,37,-2],'newroad_group_sentinel':-2,'dynamic_actor_ids':rt.dynamic_actor_ids,'delete_actor_ids_by_state':{'keep':[],'delete':[37]},
  'actual_native_joint_calls':calls,'actual_raster_kernel_forwards':calls,'backwards':0,'optimizer_updates':0,'parameters_and_source_owners_unchanged':True,
  'adapter_source_sha256':sha(source),'executed_worker_sha256':sha(Path(__file__)),'runtime_receipt':rt.receipt(),'seconds':time.time()-start,'max_reserved_cuda_bytes':int(torch.cuda.max_memory_reserved()),
  'no_v44_or_C2_or_semantic_subset_scene_loaded':True,'automatic_training':False,'automatic49':False,'quality_accepted':False,'shutdown':False,'reset_card':False}
 write(outdir/'receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['frames','runtime_receipt']}),flush=True)
if __name__=='__main__':main()
