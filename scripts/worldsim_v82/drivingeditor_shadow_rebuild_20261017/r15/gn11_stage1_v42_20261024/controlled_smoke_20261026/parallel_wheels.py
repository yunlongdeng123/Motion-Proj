"""Collect explicit-version Linux wheels from a PyPI mirror, then aria2 ranges.
No installed environment is modified by this downloader.
"""
import urllib.request,re,html,concurrent.futures,json,subprocess,sys
from urllib.parse import urljoin
from pathlib import Path
root=Path('/root/autodl-tmp/v42_wheels');root.mkdir(exist_ok=True)
versions={'torch':'2.9.1','torchvision':'0.24.1','nvidia-cuda-nvrtc-cu12':'12.8.93','nvidia-cuda-runtime-cu12':'12.8.90','nvidia-cuda-cupti-cu12':'12.8.90','nvidia-cudnn-cu12':'9.10.2.21','nvidia-cublas-cu12':'12.8.4.1','nvidia-cufft-cu12':'11.3.3.83','nvidia-curand-cu12':'10.3.9.90','nvidia-cusolver-cu12':'11.7.3.90','nvidia-cusparse-cu12':'12.5.8.93','nvidia-cusparselt-cu12':'0.7.1','nvidia-nccl-cu12':'2.27.5','nvidia-nvshmem-cu12':'3.3.20','nvidia-nvtx-cu12':'12.8.90','nvidia-nvjitlink-cu12':'12.8.93','nvidia-cufile-cu12':'1.13.1.3','triton':'3.5.1'}
if len(sys.argv)>1:
    versions=json.loads(Path(sys.argv[1]).read_text());root=Path('/root/autodl-tmp/v42_extra_wheels');root.mkdir(exist_ok=True)
def resolve(pair):
    package,version=pair
    req=urllib.request.Request('https://mirrors.aliyun.com/pypi/simple/'+package+'/',headers={'User-Agent':'pip/26'})
    data=urllib.request.urlopen(req,timeout=60).read().decode()
    urls=[urljoin(req.full_url,html.unescape(x).split('#')[0]) for x in re.findall(r'href="([^"]+)"',data)]
    ok=[]
    for url in urls:
        name=url.rsplit('/',1)[-1]
        if not name.startswith(package.replace('-','_')+'-'+version+'-'):continue
        if not name.endswith('.whl') or 'x86_64' not in name or 'manylinux' not in name:continue
        if not any(s in name for s in ['cp312-cp312','py3-none','cp39-abi3']):continue
        ok.append(url)
    if len(ok)!=1:raise ValueError((package,ok))
    print(package,flush=True);return dict(package=package,version=version,url=ok[0],filename=ok[0].rsplit('/',1)[-1])
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:rows=list(e.map(resolve,versions.items()))
(root/'manifest.json').write_text(json.dumps(rows,indent=2))
(root/'aria2.txt').write_text(''.join(r['url']+'\n  out='+r['filename']+'\n' for r in rows))
subprocess.run(['aria2c','-i',str(root/'aria2.txt'),'-d',str(root),'-j','6','-x','16','-s','16','-k','1M','--user-agent=pip/26','--continue=true','--summary-interval=15','--console-log-level=warn','--download-result=full'],check=True)
