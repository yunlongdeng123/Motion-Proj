import pathlib,json,time,requests,concurrent.futures,shutil
P=pathlib.Path('/root/autodl-tmp');D=P/'models/worldsim_v74_mainfig';D.mkdir(exist_ok=True)
url='https://huggingface.co/RainyNight/DVGT-2/resolve/main/dvgt2.pt';dest=D/'dvgt2.pt';part=dest.with_suffix('.pt.part');chunks=D/'dvgt2_chunks';chunks.mkdir(exist_ok=True)
proxy={'http':'http://127.0.0.1:35357','https':'http://127.0.0.1:35357'}
assert not dest.exists();started=time.time();start=part.stat().st_size if part.exists() else 0
with requests.get(url,proxies=proxy,stream=True,timeout=30) as q:q.raise_for_status();total=int(q.headers['Content-Length']);signed=q.url
jobs=[];size=(total-start+7)//8
for i in range(8):
 lo=start+i*size;hi=min(total,lo+size)-1
 if lo<=hi:jobs.append((i,lo,hi))
def fetch(job):
 i,lo,hi=job;p=chunks/f'{i:02d}.part';session=requests.Session();session.trust_env=False
 for attempt in range(5):
  n=p.stat().st_size if p.exists() else 0
  if n==hi-lo+1:return i
  try:
   with session.get(signed,headers={'Range':f'bytes={lo+n}-{hi}'},stream=True,timeout=(30,90)) as q:
    q.raise_for_status();assert q.status_code==206 and q.headers['Content-Range'].startswith(f'bytes {lo+n}-')
    with p.open('ab' if n else 'wb') as f:
     for data in q.iter_content(4*1024*1024):
      if data:f.write(data)
   assert p.stat().st_size==hi-lo+1;return i
  except Exception as e:
   print('segment',i,type(e).__name__,flush=True)
   if attempt==4:raise
   time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
 fs=[ex.submit(fetch,j) for j in jobs]
 while not all(f.done() for f in fs):
  done=start+sum(p.stat().st_size for p in chunks.glob('*.part'));print(json.dumps({'downloaded_GB':round(done/1e9,3),'total_GB':round(total/1e9,3),'elapsed_s':round(time.time()-started),'streams':8}),flush=True);concurrent.futures.wait(fs,timeout=15,return_when=concurrent.futures.ALL_COMPLETED)
 for f in fs:f.result()
with part.open('ab') as out:
 for i,lo,hi in jobs:
  with (chunks/f'{i:02d}.part').open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
assert part.stat().st_size==total;part.rename(dest)
record={'source':url,'user_authorized_mirror':True,'mirror_identity':'Author-hosted official checkpoint','file':str(dest),'bytes':total,'version':'official DVGT-2 public checkpoint','elapsed_s':time.time()-started,'range_streams':8}
(D/'dvgt2_download.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
