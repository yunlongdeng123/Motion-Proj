"""Official asset transfer via the verified faster local HTTPS route, then SCP."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parent/'asset_transfer'
REMOTE='/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1/assets'
FILES=[('ltf_seed_0.ckpt','autonomousvision/navsim_baselines','ltf/ltf_seed_0.ckpt',673235500),
 ('scenes/nuscenes/scene-0013.zip','datasets/XDimLab/HUGSIM','scenes/nuscenes/scene-0013.zip',444217604),
 ('scenes/nuscenes/scene-0038.zip','datasets/XDimLab/HUGSIM','scenes/nuscenes/scene-0038.zip',431724476),
 ('scenes/nuscenes/scene-0041.zip','datasets/XDimLab/HUGSIM','scenes/nuscenes/scene-0041.zip',486006755)]
CHUNK=8*1024*1024

def part(job):
    url,start,end,dest=job
    if dest.exists() and dest.stat().st_size==end-start+1:return
    for attempt in range(6):
        try:
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60) as r:
                if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Range not honored')
                data=r.read()
            if len(data)!=end-start+1:raise RuntimeError('Incomplete response')
            dest.write_bytes(data);return
        except Exception as e:
            print('RETRY',start,attempt,type(e).__name__,str(e)[:80],flush=True)
            if attempt==5:raise
            time.sleep(2)

for local,repo,path,size in FILES:
    dest=ROOT/local;dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        parts=ROOT/'parts'/local.replace('/','_');parts.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/{repo}/resolve/main/{path}'
        jobs=[(url,s,min(size-1,s+CHUNK-1),parts/f'{s:012d}.part') for s in range(0,size,CHUNK)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for i,_ in enumerate(ex.map(part,jobs)):
                if (i+1)%8==0:print(local,i+1,len(jobs),flush=True)
        with dest.open('wb') as f:
            for *_,p in jobs:f.write(p.read_bytes())
    subprocess.run(['scp',str(dest),'wm-3090-0811:'+REMOTE+'/'+local],check=True)
    print('DELIVERED',local,dest.stat().st_size,flush=True)
