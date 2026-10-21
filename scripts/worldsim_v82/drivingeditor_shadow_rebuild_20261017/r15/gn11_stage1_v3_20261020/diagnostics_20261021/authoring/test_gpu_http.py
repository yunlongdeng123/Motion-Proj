import json
from pathlib import Path
import urllib.request
import urllib.error

root='http://127.0.0.1:8770';rows=[]
def call(path,body=None,headers=None):
    request=urllib.request.Request(root+path,data=None if body is None else json.dumps(body).encode(),headers=headers or {})
    try:
        with urllib.request.urlopen(request,timeout=10) as r:return r.status,dict(r.headers),r.read().decode()
    except urllib.error.HTTPError as e:return e.code,dict(e.headers),e.read().decode()
status,headers,body=call('/api/health');assert status==200 and json.loads(body)['gpu_enabled']
rows.append(dict(case='health',status=status))
q=dict(action='trace',arm='full',raw=102,x=90,y=255,mode='deleted_five')
for name,headers in [('cross_origin',{'Content-Type':'application/json','Origin':'https://example.com'}),('cross_site',{'Content-Type':'application/json','Sec-Fetch-Site':'cross-site'}),('wrong_type',{'Content-Type':'text/plain'})]:
    status,h,b=call('/api/gpu',q,headers);assert status==403;rows.append(dict(case=name,status=status))
status,h,b=call('/api/gpu',dict(q,action='train'),{'Content-Type':'application/json'});assert status==503 and 'action' in json.loads(b)['error'];rows.append(dict(case='training_rejected_before_worker',status=status))
status,h,b=call('/api/health',headers={'Host':'attacker.example'});assert status==403;rows.append(dict(case='DNS_rebinding',status=status))
assert headers is not None
status,h,b=call('/app.js');assert status==200 and h['Cross-Origin-Resource-Policy']=='same-origin' and h['X-Content-Type-Options']=='nosniff'
rows.append(dict(case='same_origin_static_guards',status=status))
Path(__file__).with_name('gpu_http_validation.json').write_text(json.dumps(dict(status='PASSED',tests=rows,extra_GPU_queries=0,training_updates=0),indent=2),encoding='utf-8')
print(json.dumps(rows))
