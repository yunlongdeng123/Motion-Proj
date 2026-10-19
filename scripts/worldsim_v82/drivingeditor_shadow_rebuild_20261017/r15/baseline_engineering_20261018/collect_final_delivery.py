"""Download only the actual final PNG/JSON archive, verify and safely extract once."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,subprocess,tarfile,shutil
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
REMOTE='wm-vgpu-1011:/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 receipt=W/'final_full49_delivery.json'
 subprocess.run(['scp',REMOTE+receipt.name,str(receipt)],check=True)
 r=json.loads(receipt.read_text());assert r['status']=='COMPLETE_FINAL_FULL49_PNG_JSON_ARCHIVE' and not r['checkpoints_or_NPZ_included']
 archive=W/'final_full49_delivery.tar.gz'
 if archive.exists():assert archive.stat().st_size==r['bytes'] and sha(archive)==r['sha256'],'preserve and inspect mismatching existing archive'
 else:
  partial=W/'final_full49_delivery.tar.gz.partial';assert not partial.exists(),'inspect existing partial before retry'
  subprocess.run(['scp',REMOTE+archive.name,str(partial)],check=True)
  assert partial.stat().st_size==r['bytes'] and sha(partial)==r['sha256'],'preserve partial on mismatch'
  partial.replace(archive)
 manifest={row['path']:row for row in r['files']};assert len(manifest)==len(r['files'])
 with tarfile.open(archive,'r:gz') as tar:
  members=tar.getmembers();assert len(members)==len(manifest) and {m.name for m in members}==set(manifest)
  for member in members:
   rel=PurePosixPath(member.name)
   assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in ['direct_full49','all_vehicle_full49']
   assert rel.suffix in ['.png','.json']
   target=W.joinpath(*rel.parts).resolve();assert target.is_relative_to(W)
   row=manifest[member.name];assert member.size==row['bytes']
   if target.exists():assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'];continue
   target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.extracting');assert not partial.exists()
   with tar.extractfile(member) as source,partial.open('xb') as dest:shutil.copyfileobj(source,dest)
   assert sha(partial)==row['sha256'];partial.replace(target)
 for name in ['direct_full49','all_vehicle_full49']:
  data=json.loads((W/name/'receipt.json').read_text());assert data['frames']==49 and data['parameter_updates']==0
  (O/name).mkdir(exist_ok=True)
  for filename in ['receipt.json','restore_receipt.json','progress.json']:
   if (W/name/filename).exists():shutil.copy2(W/name/filename,O/name/filename)
 shutil.copy2(receipt,O/receipt.name)
 record=dict(status='VERIFIED_ACTUAL_FINAL_ARCHIVE_SAFELY_EXTRACTED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_sha256=r['sha256'],bytes=r['bytes'],verified_files=len(manifest),all_member_hashes_verified=True,only_PNG_JSON=True,training_updates=0,quality_verdict_not_assigned=True)
 for base in [W,O]:(base/'final_delivery_collection.json').write_text(json.dumps(record,indent=2))
 print(json.dumps(record))
if __name__=='__main__':main()
