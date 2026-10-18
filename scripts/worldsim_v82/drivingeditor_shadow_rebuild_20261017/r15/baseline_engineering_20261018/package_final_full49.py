"""Package only completed full-frame PNG/JSON evaluation outputs, CPU-only."""
from pathlib import Path
import datetime,hashlib,json,re,tarfile
from PIL import Image
E=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 archive=E/'final_full49_delivery.tar.gz';receipt=E/'final_full49_delivery.json'
 if receipt.exists():
  r=read(receipt);assert r['status']=='COMPLETE_FINAL_FULL49_PNG_JSON_ARCHIVE' and archive.is_file() and archive.stat().st_size==r['bytes'] and sha(archive)==r['sha256']
  print(json.dumps(r));return
 assert read(E/'parallel_queue_status.json')['state']=='PARALLEL_A_B_FULL49_COMPLETE_AWAITING_REPORT_GIT','wait for both final evaluations, do not package partial results'
 frames={f'0008{n:03d}' for n in range(89,138)};actors={6,37,5,26,18}
 base={'gt.png','full.png','dynamic.png','static.png','dynamic_alpha.png','static_alpha.png','delete_all.png'}
 groups=[];files=[];stats={}
 for name in ['direct_full49','all_vehicle_full49']:
  root=E/name;r=read(root/'receipt.json');assert r['frames']==49 and r['parameter_updates']==0
  if name=='direct_full49':
   assert set(r['actors'])==actors and len(r['records'])==245
   assert {(x['frame_id'],x['actor']) for x in r['records']}=={(f,a) for f in frames for a in actors}
   channels=base|{f'a{a:02d}_{n}.png' for a in actors for n in ['delete','dynamic_delete','dynamic_delete_alpha','isolated','alpha']}
  else:
   assert len(r['records'])==49 and {x['frame_id'] for x in r['records']}==frames;channels=base
  got={p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r'0008[0-9]{3}',p.name)};assert got==frames
  for frame in sorted(frames):
   directory=root/frame;pngs={p.name for p in directory.glob('*.png')};assert pngs==channels,(name,frame,pngs^channels)
   for channel in sorted(channels):
    p=directory/channel;assert not p.is_symlink() and p.resolve().parent==directory.resolve()
    with Image.open(p) as image:assert image.size==(480,320);image.verify()
    files.append(p)
  for n in ['receipt.json','restore_receipt.json','progress.json']:
   if (root/n).is_file():read(root/n);files.append(root/n)
  stats[name]=dict(frames=49,PNG_channels_per_frame=len(channels),PNG_files=len(channels)*49,records=len(r['records']))
 before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files};partial=archive.with_suffix('.gz.partial')
 assert not archive.exists() and not partial.exists(),'inspect any prior partial/archive before retrying; never overwrite silently'
 manifest=[dict(path=p.relative_to(E).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
 with tarfile.open(partial,'w:gz',compresslevel=1) as tar:
  for p in files:tar.add(p,arcname=p.relative_to(E).as_posix(),recursive=False)
 assert before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
 partial.replace(archive)
 r=dict(status='COMPLETE_FINAL_FULL49_PNG_JSON_ARCHIVE',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),groups=stats,files=manifest,
  checkpoints_or_NPZ_included=False,source_files_unchanged=True,all_images_full_frame=True,GPU_used=False,training_updates=0,quality_accepted=False,human_verdict=None)
 receipt.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='files'}))
if __name__=='__main__':main()
