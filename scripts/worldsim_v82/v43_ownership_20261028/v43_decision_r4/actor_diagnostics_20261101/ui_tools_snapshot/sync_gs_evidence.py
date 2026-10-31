from pathlib import Path,PurePosixPath
import argparse,hashlib,json,os,shutil,tempfile,math
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,action='append',required=True);a=p.parse_args()
source=a.source;js=source/'gs_evidence.js';d=json.loads(js.read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'));images=set();pairs=0;actors=0
for context,tracks in d.items():
 for track,b in tracks.items():
  actors+=1
  if b.get('source_context')!=context or b.get('formal_f_geom') is not None:raise ValueError('invalid geometry provenance')
  for f in b['frames']:
   pairs+=1;xyz=f['xyz'];lid=f['lidar_xyz'];n=len(xyz)
   if f['context_id']!=context or f['track_id']!=track or f.get('formal_f_geom') is not None or f.get('formal_gate') is not None or f['coordinates']!='actor_local_x_forward_y_left_z_up_box_center' or len(f['source_branch'])!=n or len(f['original_row_id'])!=n or len(f['effective_opacity'])!=n or any(x not in ('static','dynamic') for x in f['source_branch']):raise ValueError('invalid geometry structure')
   if any(len(point)!=3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in point) for point in xyz+lid):raise ValueError('invalid coordinates')
   rel=PurePosixPath(f['image']);
   if rel.is_absolute() or '..' in rel.parts or len(rel.parts)!=2 or rel.parts[0]!='geometry':raise ValueError('invalid image path')
   file=source/rel.name
   if not file.is_file():raise FileNotFoundError(file)
   images.add(file)
for target in a.target:
 data=json.loads((target/'data.js').read_text(encoding='utf8').split('=',1)[1].strip().removesuffix(';'));registry={x['context_id']:{r['track_id'] for r in x['actors']} for x in data}
 if any(ctx not in registry or not set(tracks).issubset(registry[ctx]) for ctx,tracks in d.items()):raise ValueError('GS actor is not in target registry')
 dest=target/'geometry';dest.mkdir(parents=True,exist_ok=True)
 for f in sorted(images):shutil.copy2(f,dest/f.name)
 tmp=dest/'gs_evidence.js.tmp';shutil.copy2(js,tmp)
 if hashlib.sha256(tmp.read_bytes()).digest()!=hashlib.sha256(js.read_bytes()).digest():raise ValueError('source changed')
 os.replace(tmp,dest/'gs_evidence.js');print('SYNC',target,'contexts',len(d),'actors',actors,'frame_pairs',pairs,'images',len(images),flush=True)
