"""Download only the official assets needed by the frozen initial simulator cohort."""
import concurrent.futures
import json
import os
from pathlib import Path
import time
import requests

ROOT = Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
ASSETS = ROOT / 'assets'
FILES = [(f'scenes/nuscenes/scene-{s}.zip', 'datasets/XDimLab/HUGSIM', f'scenes/nuscenes/scene-{s}.zip') for s in ['0013','0038','0041']]
FILES += [('scenarios.zip','datasets/XDimLab/HUGSIM','scenarios.zip'), ('nusc_map_cache.zip','datasets/XDimLab/HUGSIM','nusc_map_cache.zip'), ('ltf_seed_0.ckpt','autonomousvision/navsim_baselines','ltf/ltf_seed_0.ckpt')]

def one(item):
    local,repo,remote=item
    dest=ASSETS/local
    dest.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://huggingface.co/{repo}/resolve/main/{remote}'
    if dest.exists():
        print('EXISTS',local,dest.stat().st_size,flush=True)
        return {'path':str(dest),'bytes':dest.stat().st_size,'url':url}
    tmp=dest.with_suffix(dest.suffix+'.partial')
    for attempt in range(3):
        offset=tmp.stat().st_size if tmp.exists() else 0
        try:
            with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,60)) as r:
                r.raise_for_status()
                resume=offset>0 and r.status_code==206
                with tmp.open('ab' if resume else 'wb') as f:
                    for b in r.iter_content(8*1024*1024):f.write(b)
            tmp.rename(dest)
            print('DONE',local,dest.stat().st_size,flush=True)
            return {'path':str(dest),'bytes':dest.stat().st_size,'url':url}
        except Exception as e:
            print('RETRY',local,attempt,type(e).__name__,str(e)[:200],flush=True)
            if attempt==2:raise
            time.sleep(2)

if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,FILES))
    (ROOT/'asset_downloads.json').write_text(json.dumps(results,indent=2))
