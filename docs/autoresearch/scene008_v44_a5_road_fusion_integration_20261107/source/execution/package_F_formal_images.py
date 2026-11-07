from pathlib import Path
import hashlib,json,tarfile
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_v44_a5_road_fusion_integration_20261107_r1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=json.loads((R/'F_formal400_r1/returned.json').read_text())
assert result['actual_optimizer_updates']==400 and result['actual_native_forwards']==808
assert result['status']=='COMPLETE_FIXED_400_PAIRED_BOUNDED_COLOUR_RESIDUAL_BAKE_PENDING_GATES'
files=[]
for raw in [94,105,128,137]:
 for state in ['keep','delete']:
  rel=f'F_formal400_r1/evaluation/frames/{raw:03d}/{state}.png';p=R/rel;assert p.is_file()
  files.append(dict(path='evidence/'+rel,remote_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
D=R/'delivery';D.mkdir(exist_ok=True)
manifest=D/'F_formal400_images_manifest.json';assert not manifest.exists()
manifest.write_text(json.dumps(dict(files=files,actual_candidate_colour_updates=400,training_Gaussian_updates=0),indent=2)+'\n')
archive=D/'F_formal400_images.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for f in files:t.add(f['remote_path'],arcname=f['path'],recursive=False)
 t.add(manifest,arcname='evidence/F_formal400_images_manifest.json',recursive=False)
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files))
(D/'F_formal400_images_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
