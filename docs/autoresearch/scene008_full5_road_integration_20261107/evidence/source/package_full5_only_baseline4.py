from pathlib import Path
import hashlib,json,os,tarfile
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=B/'scene008_full5_road_integration_20261107_r1';FULL=B/'scene008_full_dynamic_edit_20261107_r1';CASE=R/'baseline4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
receipt=json.loads((CASE/'receipt.json').read_text());assert receipt['actual_raster_kernel_forwards']==24 and receipt['optimizer_updates']==0
headers=[]
for frame in receipt['frames']:
 a=frame['states']['keep'];b=frame['states']['delete'];assert a['K']==b['K'] and a['W_model_to_camera']==b['W_model_to_camera'] and a['timestamp']==b['timestamp']
 k=a['K'];K=[[k['fx'],0.,k['cx']],[0.,k['fy'],k['cy']],[0.,0.,1.]]
 headers.append({'raw':frame['raw'],'K':K,'model_to_camera':a['W_model_to_camera'],'model_time':a['timestamp'],'render_wh':[k['width'],k['height']],
  'group_actor_ids':receipt['group_actor_ids'],'world_to_model_scale':receipt['model_to_world_scale'],'depth_m_rule':'depth_sum / max(alpha,1e-8) / world_to_model_scale'})
write(CASE/'frame_headers.json',{'frames':headers,'source_receipt_sha256':sha(CASE/'receipt.json'),'scale_field_clarification':'The executed receipt used the name model_to_world_scale for norm(world_to_model[:3,:3]); correct semantic is world_to_model_scale. Numerical depth conversion uses its reciprocal and is unchanged.'})
cached=[]
for raw in receipt['raw_ids']:
 p=CASE/f'frames/{raw:03d}/keep.png';candidates=[FULL/f'formal10k_keep_all49/frames/{raw:03d}/keep.png',FULL/f'formal10k_keep_all49/frames/{raw:03d}/keep.png']
 old=next((x for x in candidates if x.is_file()),None)
 if old is None:cached.append({'raw':raw,'status':'CACHED_NATIVE_PNG_NOT_FOUND_REMOTE'});continue
 equal=p.read_bytes()==old.read_bytes();assert equal,(p,old)
 cached.append({'raw':raw,'status':'ENCODED_PNG_EXACT','baseline':str(p),'cached_native':str(old),'sha256':sha(p)})
write(CASE/'cached_native_keep_replay.json',{'status':'PASS_CACHED_PNGS_EXACT' if all(x['status']=='ENCODED_PNG_EXACT' for x in cached) else 'REMOTE_CACHE_PARTIALLY_UNAVAILABLE_NO_NEW_RENDER','rows':cached,'new_native_calls':0})
files=[CASE/'receipt.json',CASE/'frame_headers.json',CASE/'cached_native_keep_replay.json',R/'contract/full5_only_baseline4_protocol_v1.json',R/'contract/baseline4_launch.json',R/'contract/full5_transform_matrix.txt',R/'source/render_full5_only_baseline4.py']
for raw in receipt['raw_ids']:
 files.append(CASE/f'frames/{raw:03d}/input_guard.json')
 for state in ['keep','delete']:files.append(CASE/f'frames/{raw:03d}/{state}.png')
D=R/'delivery';D.mkdir(exist_ok=True)
entries=[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
manifest=D/'baseline4_light_manifest.json';assert not manifest.exists();write(manifest,{'files':entries,'scope':'One Full5 scene, no v44 models','native_logical_joint_calls':8,'actual_native_kernel_forwards':24,'backwards':0,'optimizer_updates':0})
archive=D/'baseline4_light.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for p in files:t.add(p,arcname='evidence/'+str(p.relative_to(R)),recursive=False)
 t.add(manifest,arcname='evidence/baseline4_light_manifest.json',recursive=False)
proof={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files)+1,'native_logical_joint_calls':8,'actual_native_kernel_forwards':24,'optimizer_updates':0}
write(D/'baseline4_light_archive_receipt.json',proof);print(json.dumps(proof),flush=True)
