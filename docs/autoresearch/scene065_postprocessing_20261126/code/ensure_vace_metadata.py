"""Complete the small tokenizer files omitted by the original weight-only download."""
import json
import urllib.request
from pathlib import Path

root=Path('/root/autodl-tmp/models/Wan2.1-VACE-1.3B/google/umt5-xxl')
root.mkdir(parents=True,exist_ok=True)
base='https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-1.3B/resolve/master/google/umt5-xxl/'
for name in ('tokenizer_config.json','tokenizer.json','special_tokens_map.json','spiece.model'):
    path=root/name
    if path.exists() and path.stat().st_size:continue
    request=urllib.request.Request(base+name,headers={'User-Agent':'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(request,timeout=60) as response:
            data=response.read(64*1024*1024)
        assert data and len(data)<64*1024*1024
        if name.endswith('.json'):json.loads(data)
        temporary=path.with_suffix(path.suffix+'.part')
        temporary.write_bytes(data);temporary.replace(path)
        print(name,len(data),flush=True)
    except Exception as exc:
        print(name,type(exc).__name__,str(exc),flush=True)
        if name in ('tokenizer_config.json','tokenizer.json'):raise
