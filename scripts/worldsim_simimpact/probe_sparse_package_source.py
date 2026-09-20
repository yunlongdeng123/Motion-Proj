"""读取PyPI公开元数据，比较已知软件包源的有限传输，不更换软件版本。"""
import json,time,urllib.request
from pathlib import Path
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-SPARSE-NATIVE-01/20260915-r1')
metadata=json.load(urllib.request.urlopen('https://pypi.org/pypi/torch/1.13.1/json',timeout=20))
asset=next(x for x in metadata['urls'] if x['filename']=='torch-1.13.1-cp39-cp39-manylinux1_x86_64.whl')
start=time.time();req=urllib.request.Request(asset['url'],headers={'Range':'bytes=0-1048575'})
with urllib.request.urlopen(req,timeout=30) as f:
    status=f.status;content_range=f.headers.get('Content-Range');data=f.read(1048576)
result={'filename':asset['filename'],'url':asset['url'],'size':asset['size'],'requires_dist':metadata['info']['requires_dist'],'sample_bytes':len(data),'sample_seconds':time.time()-start,'status':status,'content_range':content_range}
(R/'pypi_transport_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
