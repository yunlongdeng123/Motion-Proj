"""从保留的 AutoDL 公共原包按场景恢复 V4 sweeps；默认只检查，不写文件。"""
import argparse,json,os,shutil,tarfile
from pathlib import Path

DEFAULT=Path('/root/autodl-tmp/data/worldsim_v4')
PUBLIC=Path('/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval')

def entries(root,scene):
 manifest=root/'manifests'/f'{scene}_raw_manifest_v4.json'
 if not manifest.is_file():
  manifest=Path('/root/autodl-tmp/cleanup_manifests/20260922-v4-retire/manifests')/manifest.name
 rows=json.loads(manifest.read_text())['files']
 return [r for r in rows if r['filename'].startswith('sweeps/')]

def restore(rows,output,execute=False,limit=None):
 groups={}
 for row in rows:
  rel=Path(row['filename']);assert rel.parts[0]=='sweeps' and not rel.is_absolute() and '..' not in rel.parts
  shard=row['shard'];assert Path(shard).name==shard
  src=PUBLIC/shard;assert src.is_file() and src.stat().st_size>0
  dest=output/rel
  if dest.exists():
   assert dest.is_file() and dest.stat().st_size==row['bytes'],dest
   continue
  groups.setdefault(shard,{})[str(rel)]=row
 expected=sum(len(g) for g in groups.values());written=[]
 if execute:
  for shard,wanted in groups.items():
   with tarfile.open(PUBLIC/shard,'r|gz') as archive:
    for member in archive:
     name=member.name.removeprefix('./')
     if name not in wanted:continue
     row=wanted.pop(name);assert member.isfile() and member.size==row['bytes']
     dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
     assert dest.resolve().is_relative_to(output.resolve()) and not dest.exists()
     tmp=dest.with_name(dest.name+'.restoring');assert not tmp.exists()
     with archive.extractfile(member) as src,tmp.open('xb') as dst:
      shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
     assert tmp.stat().st_size==member.size
     os.replace(tmp,dest);written.append(str(dest))
     print(json.dumps({'restored':len(written),'file':str(dest)}),flush=True)
     if limit and len(written)>=limit:return {'expected':expected,'restored':written,'limited_probe':True}
     if not wanted:break
   assert not wanted,{'missing':list(wanted)[:10],'shard':shard}
 return {'expected':expected,'restored':written,'execute':execute,'shards':list(groups)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene',action='append',required=True)
 ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--output',type=Path)
 ap.add_argument('--execute',action='store_true');ap.add_argument('--probe-one',action='store_true')
 args=ap.parse_args();rows=[]
 for scene in args.scene:rows+=entries(args.root,scene)
 print(json.dumps(restore(rows,args.output or args.root/'drivestudio_raw_trainval',args.execute,1 if args.probe_one else None)),flush=True)
if __name__=='__main__':main()
