"""Read-only digest verification of every file in the original frozen manifest."""
import datetime
import hashlib
import json
from pathlib import Path
import time

camp=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_serial_r1_20261021')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=camp/'final49/summary_r2/asset_manifest.json'
assert sha(manifest)=='eb926ef96b21f5b7382bc7e9eff7f4424cb13fc914246482d90ad179f6b72465'
files=json.loads(manifest.read_text())['files'];errors=[];total=0;started=time.monotonic()
for index,(name,expected) in enumerate(files.items()):
    p=camp/name
    if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha(p)!=expected['sha256']:errors.append(name)
    total+=expected['bytes']
    if index%1000==0:print(json.dumps(dict(checked=index,errors=len(errors))),flush=True)
receipt=dict(status='ORIGINAL_FROZEN_MANIFEST_ALL_SHA256_UNCHANGED' if not errors else 'AUDIT_MISMATCH',
    files=len(files),bytes=total,errors=errors,elapsed_s=time.monotonic()-started,optimizer_updates=0,
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
output=Path(__file__).resolve().parent/('original_asset_audit_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
with output.open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(dict(receipt,path=str(output))),flush=True)
if errors:raise SystemExit(1)
