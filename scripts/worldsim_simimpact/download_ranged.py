"""Bounded range transfers to survive the observed proxy's long-response disconnects."""
import concurrent.futures
import json
from pathlib import Path
import requests
import time

ROOT=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
ASSETS=ROOT/'assets'
CHUNK=8*1024*1024
# Prioritize the first runnable simulator/policy pair. All initial scenes stay registered.
FILES=[('scenarios.zip','datasets/XDimLab/HUGSIM','scenarios.zip',203858),
 ('ltf_seed_0.ckpt','autonomousvision/navsim_baselines','ltf/ltf_seed_0.ckpt',673235500),
 ('scenes/nuscenes/scene-0013.zip','datasets/XDimLab/HUGSIM','scenes/nuscenes/scene-0013.zip',444217604),
 ('scenes/nuscenes/scene-0038.zip','datasets/XDimLab/HUGSIM','scenes/nuscenes/scene-0038.zip',431724476),
 ('scenes/nuscenes/scene-0041.zip','datasets/XDimLab/HUGSIM','scenes/nuscenes/scene-0041.zip',486006755)]

def get_part(job):
    url,start,end,dest=job
    expected=end-start+1
    if dest.exists() and dest.stat().st_size==expected:return
    for attempt in range(6):
        try:
            with requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(20,40)) as r:
                r.raise_for_status()
                if r.status_code!=206 and not(start==0 and len(r.content)==expected):
                    raise RuntimeError(f'Range not honored: {r.status_code}')
                data=r.content
                if len(data)!=expected:raise RuntimeError(f'Short response {len(data)}/{expected}')
                if r.status_code==206 and not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Wrong range')
            dest.write_bytes(data)
            return
        except Exception as e:
            if attempt==5:raise
            print('retry',dest.name,attempt,type(e).__name__,flush=True);time.sleep(min(2**attempt,8))

def main():
    results=[]
    for local,repo,path,size in FILES:
        dest=ASSETS/local;dest.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/{repo}/resolve/main/{path}'
        if dest.exists() and dest.stat().st_size==size:
            results.append({'path':str(dest),'url':url,'bytes':size});continue
        parts=ASSETS/'range_parts'/local.replace('/','_');parts.mkdir(parents=True,exist_ok=True)
        existing=dest.with_suffix(dest.suffix+'.partial')
        jobs=[]
        for start in range(0,size,CHUNK):
            end=min(size-1,start+CHUNK-1);part=parts/f'{start:012d}.part'
            if not part.exists() and existing.exists() and existing.stat().st_size>=end+1:
                with existing.open('rb') as f:f.seek(start);part.write_bytes(f.read(end-start+1))
            jobs.append((url,start,end,part))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            done=0
            for _ in ex.map(get_part,jobs):
                done+=1
                if done%8==0:print(local,done,'/',len(jobs),flush=True)
        tmp=dest.with_suffix(dest.suffix+'.assembled')
        with tmp.open('wb') as f:
            for _,_,_,part in jobs:f.write(part.read_bytes())
        tmp.rename(dest)
        results.append({'path':str(dest),'url':url,'bytes':size})
        (ROOT/'asset_downloads_ranged.json').write_text(json.dumps(results,indent=2))
        print('DONE',local,size,flush=True)

if __name__=='__main__':main()
