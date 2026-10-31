from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile
p=argparse.ArgumentParser(description='One-shot sync of completed media bundles; media.js is published last.')
p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,action='append',required=True);a=p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def parse_script(p):
 s=p.read_text(encoding='utf8');i=s.find(']={');
 if i<0:raise ValueError('Unexpected media.js envelope')
 value,end=json.JSONDecoder().raw_decode(s[i+2:]);return value
def safe_ref(context,value):
 rel=PurePosixPath(value)
 if rel.is_absolute() or '..' in rel.parts or tuple(rel.parts[:2])!=('scenes',context):raise ValueError('Invalid media reference '+str(value))
 return Path(*rel.parts[2:])
def copy_checked(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() and source.stat().st_size==dest.stat().st_size and sha(source)==sha(dest):return False
 fd,name=tempfile.mkstemp(prefix='.r4-sync-',dir=dest.parent);os.close(fd);tmp=Path(name)
 try:
  shutil.copyfile(source,tmp)
  if sha(source)!=sha(tmp):raise ValueError('Source changed while copying '+str(source))
  os.replace(tmp,dest)
 finally:
  if tmp.exists():tmp.unlink()
 return True
for scene in sorted(a.source.iterdir()):
 if not scene.is_dir():continue
 try:
  manifest=scene/'manifest.json';points=scene/'frame_point_views.json';media=scene/'media.json';js=scene/'media.js'
  if not all(x.is_file() for x in [manifest,points,media,js]):print('SKIP',scene.name,'not complete');continue
  m=json.loads(manifest.read_text(encoding='utf8'));audit=json.loads(points.read_text(encoding='utf8'))
  if m.get('LiDAR_status')!='decoded' or not m.get('full_sequence') or not audit.get('nonzero_measured_frames_never_empty') or not audit.get('aggregate_subset_not_reused'):print('SKIP',scene.name,'not marked decoded/complete');continue
  d=json.loads(media.read_text(encoding='utf8')); context=scene.name
  if m['context_id']!=context or d['context_id']!=context or d['schema']!='actor_media_v1' or parse_script(js)!=d:raise ValueError('media identity or JS/JSON mismatch')
  n=d['frame_count'];actors=d['actors'];frames=d['frames']
  if n!=m['raw_frame_count'] or n!=m['expected_frames'] or n!=d['full_raw_frame_denominator'] or len(frames)!=n or [x['raw'] for x in frames]!=list(range(n)) or len(actors)!=m['actors'] or len(actors)!=audit['actors']:raise ValueError('incomplete frame or actor denominator')
  refs=set()
  for f in frames:refs.add(safe_ref(context,f['image']))
  for k in ['video','box_video']:refs.add(safe_ref(context,d[k]))
  for track,actor in actors.items():
   lid=actor.get('lidar') or {}
   if not lid.get('data_js') or not lid.get('frame_data_js') or not lid.get('npz'):raise ValueError('missing per-actor LiDAR data '+track)
   for key in ['data_js','frame_data_js','preview','npz']:
    if lid.get(key):refs.add(safe_ref(context,lid[key]))
  files=[scene/r for r in sorted(refs)]
  if not all(f.is_file() for f in files):raise FileNotFoundError('missing '+str(next(f for f in files if not f.is_file())))
  for v in ['video','box_video']:
   probe=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',str(scene/safe_ref(context,d[v]))],capture_output=True,text=True,check=True).stdout)['streams'][0]
   if int(probe.get('nb_frames',-1))!=n or [probe.get('width'),probe.get('height')]!=d['imageWH']:raise ValueError('video frame count or dimensions mismatch')
  for target in a.target:
   reg=(target/'data.js').read_text(encoding='utf8');catalog=json.loads(reg.split('=',1)[1].strip().removesuffix(';'));entry=next((x for x in catalog if x['context_id']==context),None)
   if not entry or {x['track_id'] for x in entry['actors']}!=set(actors):raise ValueError('actor inventory differs for '+str(target))
  snapshot=sha(js)
  for target in a.target:
   dst=target/'scenes'/context;changed=0
   for source in files+[media,manifest,points]:changed+=copy_checked(source,dst/source.relative_to(scene))
   if sha(js)!=snapshot:raise ValueError('media.js changed during transfer')
   changed+=copy_checked(js,dst/'media.js');print('SYNC',context,'to',str(target),'frames',n,'actors',len(actors),'changed',changed,flush=True)
 except Exception as e:
  print('ERROR',scene.name,str(e),file=sys.stderr,flush=True);raise
