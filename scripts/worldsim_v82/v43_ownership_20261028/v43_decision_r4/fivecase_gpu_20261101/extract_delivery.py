from pathlib import Path
import hashlib,tarfile,json
w=Path('work/r4_five_gpu_20261101');a=w/'delivery_portable.tar.gz'
with a.open('wb') as out:
 for i in range(4):
  p=w/f'portable_part{i}.bin';assert p.stat().st_size==[88684714,88684714,88684714,88684712][i]
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):out.write(b)
h=hashlib.sha256(a.read_bytes()).hexdigest();assert h=='f645161c14cf62a73f8781c42ab2a7ebaa59c774f56208a52f8e189673952e71',h
out=Path('outputs/V43_R4_FiveCase_GPU_20261101').resolve();out.mkdir(parents=True,exist_ok=True)
with tarfile.open(a) as t:
 for m in t.getmembers():
  r=Path(m.name);assert r.parts[0]=='delivery' and '..' not in r.parts
  assert not m.issym() and not m.islnk()
  if len(r.parts)==1:continue
  q=out.joinpath(*r.parts[1:]);assert q.is_relative_to(out)
  if m.isdir():q.mkdir(parents=True,exist_ok=True)
  elif m.isfile():
   q.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as src,q.open('wb') as dst:
    for b in iter(lambda:src.read(8*1024*1024),b''):dst.write(b)
(out/'retrieval_receipt.json').write_text(json.dumps(dict(archive_sha256=h,archive_bytes=a.stat().st_size,verified=True,scope=json.loads((out/'portable_scope.json').read_text())),ensure_ascii=False,indent=2),encoding='utf-8')
print('EXTRACTED',len(list(out.rglob('*'))))
