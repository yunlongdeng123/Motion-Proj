import pathlib,io,zipfile,json,requests,struct,zlib,concurrent.futures,time
D=pathlib.Path('/root/autodl-tmp/models/worldsim_v74_mainfig');out=D/'noksr';out.mkdir(exist_ok=True)
url='https://drive.usercontent.google.com/download?id=14Xrbbox87tLjwE4vmQkEM2u5-vrWW008&export=download&confirm=t';proxy={'http':'http://127.0.0.1:35357','https':'http://127.0.0.1:35357'}
with requests.get(url,headers={'Range':'bytes=-65536'},proxies=proxy,timeout=60) as resp:
 resp.raise_for_status();assert resp.status_code==206;total=int(resp.headers['Content-Range'].split('/')[-1]);tail=resp.content
base=total-len(tail)
class TailFile(io.RawIOBase):
 def __init__(self):self.pos=0
 def seek(self,x,w=0):self.pos=x if w==0 else (self.pos+x if w==1 else total+x);return self.pos
 def tell(self):return self.pos
 def read(self,n=-1):
  if n<0:n=total-self.pos
  assert self.pos>=base
  data=tail[self.pos-base:self.pos-base+n];self.pos+=len(data);return data
 def seekable(self):return True
z=zipfile.ZipFile(TailFile());infos=z.infolist();(out/'archive_index.json').write_text(json.dumps([{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size,'offset':x.header_offset} for x in infos],indent=2));print([x.filename for x in infos],flush=True)
info=next(x for x in infos if x.filename.endswith('/Carla_Serial_best.ckpt'));dest=out/'Carla_Serial_best.ckpt'
if dest.exists():print('ALREADY_READY');raise SystemExit(0)
def get(lo,hi):
 for attempt in range(5):
  try:
   r=requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},proxies=proxy,timeout=(20,120));r.raise_for_status();assert r.status_code==206 and len(r.content)==hi-lo+1;return r.content
  except Exception:
   if attempt==4:raise
   time.sleep(1)
h=get(info.header_offset,info.header_offset+29);q=struct.unpack('<4s5H3I2H',h);offset=info.header_offset+30+q[-1]+q[-2];chunks=out/'carla_serial_compressed';chunks.mkdir(exist_ok=True);size=(info.compress_size+7)//8
def one(i):
 lo=i*size;hi=min(info.compress_size,lo+size)-1;p=chunks/f'{i:02d}.part'
 if hi<lo:return
 if p.exists() and p.stat().st_size==hi-lo+1:return
 p.write_bytes(get(offset+lo,offset+hi));print('SEGMENT_DONE',i,p.stat().st_size,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(one,range(8)))
dec=zlib.decompressobj(-15);part=dest.with_suffix('.ckpt.part')
with part.open('wb') as f:
 for i in range(8):
  with (chunks/f'{i:02d}.part').open('rb') as src:
   while b:=src.read(2**20):f.write(dec.decompress(b))
assert dec.eof and part.stat().st_size==info.file_size;part.rename(dest)
(out/'download.json').write_text(json.dumps({'source':'official NoKSR README Google Drive archive','url':url,'member':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size,'strategy':'Read ZIP central directory, download only selected member in 8 ranges; no checkpoint modification'},indent=2));print('CARLA_SERIAL_READY',dest,flush=True)
