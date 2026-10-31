from pathlib import Path,PurePosixPath
import argparse,hashlib,json,sys
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def hash_file(f):
 h=hashlib.sha256()
 with f.open('rb') as x:
  for b in iter(lambda:x.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rel(context,value):
 p=PurePosixPath(value)
 if p.is_absolute() or '..' in p.parts or tuple(p.parts[:2])!=('scenes',context):raise ValueError('unsafe reference '+str(value))
 return Path(*p.parts[2:])
def data_js(path):return json.loads(path.read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'))
catalog=data_js(a.target[0]/'data.js');contexts={x['context_id']:{r['track_id'] for r in x['actors']} for x in catalog};assert len(contexts)==15 and sum(map(len,contexts.values()))==1228
for dest in a.target[1:]:
 r={x['context_id']:{z['track_id'] for z in x['actors']} for x in data_js(dest/'data.js')}
 if r!=contexts:raise ValueError('Target actor registries differ')
missing_scenes=[];scenes=[];missing=[];mismatch=[]
for context,expected in contexts.items():
 source=a.source/context;mf=source/'manifest.json';pj=source/'frame_point_views.json';mj=source/'media.json';js=source/'media.js'
 if not all(x.is_file() for x in [mf,pj,mj,js]):missing_scenes.append(context);continue
 manifest=json.loads(mf.read_text(encoding='utf8'));audit=json.loads(pj.read_text(encoding='utf8'));m=json.loads(mj.read_text(encoding='utf8'))
 if manifest.get('LiDAR_status')!='decoded' or not manifest.get('full_sequence') or not audit.get('nonzero_measured_frames_never_empty') or not audit.get('aggregate_subset_not_reused'):missing_scenes.append(context);continue
 if m['context_id']!=context or set(m['actors'])!=expected or m['frame_count']!=m['full_raw_frame_denominator'] or len(m['frames'])!=m['frame_count'] or [f['raw'] for f in m['frames']]!=list(range(m['frame_count'])):raise ValueError('source denominator mismatch '+context)
 refs={Path('manifest.json'),Path('frame_point_views.json'),Path('media.json'),Path('media.js')}
 refs.update(rel(context,f['image']) for f in m['frames'])
 for k in ['video','box_video']:refs.add(rel(context,m[k]))
 for track,v in m['actors'].items():
  l=v.get('lidar') or {}
  for k in ['data_js','frame_data_js','preview','npz']:
   if not l.get(k):raise ValueError('missing manifest reference '+context+' '+track+' '+k)
   refs.add(rel(context,l[k]))
 files=[]
 for r in sorted(refs):
  f=source/r
  if not f.is_file():missing.append({'location':'source','context':context,'file':r.as_posix()});continue
  digest=hash_file(f);size=f.stat().st_size;entry={'path':r.as_posix(),'bytes':size,'sha256':digest}
  for dest in a.target:
   target=dest/'scenes'/context/r
   if not target.is_file():missing.append({'location':str(dest),'context':context,'file':r.as_posix()})
   elif target.stat().st_size!=size or hash_file(target)!=digest:mismatch.append({'location':str(dest),'context':context,'file':r.as_posix()})
  files.append(entry)
 scenes.append({'scene_key':m['scene_key'],'context_id':context,'frames':m['frame_count'],'actors':len(m['actors']),'file_count':len(files),'files':files})
 print('CHECKED',m['scene_key'],'frames',m['frame_count'],'actors',len(m['actors']),'refs',len(refs),flush=True)
result={'schema':'actor_media_sync_receipt_v1','source':str(a.source),'targets':[str(x) for x in a.target],'expected_scenes':15,'expected_actors':1228,'expected_frames':2975,'complete':len(scenes)==15 and sum(x['actors'] for x in scenes)==1228 and sum(x['frames'] for x in scenes)==2975 and not missing_scenes and not missing and not mismatch,'scene_count':len(scenes),'actor_count':sum(x['actors'] for x in scenes),'frame_count':sum(x['frames'] for x in scenes),'file_count_per_target':sum(x['file_count'] for x in scenes),'missing_scenes':missing_scenes,'missing_references':missing,'content_mismatches':mismatch,'scene_files':scenes,'notes':['Only full decoded scenes with independently sampled per-frame data are counted.','Historical diagnostic backups are not copied.','Browser interactions are not verified by this receipt.']}
if not result['complete']:
 print('INCOMPLETE',json.dumps({'scene_count':len(scenes),'missing_scenes':missing_scenes,'missing_count':len(missing),'mismatch_count':len(mismatch)}),file=sys.stderr);sys.exit(2)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('RECEIPT',a.output,flush=True)
