import json,hashlib,tarfile,shutil
from pathlib import Path
import numpy as np
from segmentation import ROOT,CONTEXTS,write
from export_diagnostics import pack
dest=ROOT/'depth_delivery_v1';dest.mkdir(exist_ok=False);records=[]
for ctx in CONTEXTS:
    source=ROOT/'depth_support_v1'/ctx;out=dest/ctx;out.mkdir();summary=json.loads((source/'summary.json').read_text());index=dict(frames={},summary=summary)
    for r in summary['records']:
        raw=r['raw'];d=np.load(source/('%03d.npz'%raw));packed=pack({k:d[k] for k in d.files});name='depth_%03d.js'%raw
        (out/name).write_text('window.GPU_DEPTH_FRAMES=window.GPU_DEPTH_FRAMES||{};window.GPU_DEPTH_FRAMES['+str(raw)+']='+json.dumps(packed,separators=(',',':'))+';');index['frames'][str(raw)]='gpu_semantics/'+name
    (out/'depth_index.js').write_text('window.GPU_DEPTH_INDEX='+json.dumps(index,separators=(',',':'))+';');write(out/'depth_summary.json',summary)
    records.append(dict(context=ctx,static_repeated_compatible_rows=summary['static_repeated_compatible_rows'],dynamic_repeated_compatible_rows=summary['dynamic_repeated_compatible_rows'],static_return_mass=sum(r['static_sparse_return_alpha_mass'] for r in summary['records']),static_compatible_mass=sum(r['static_depth_compatible_mass'] for r in summary['records']),static_farther_than_1m_mass=sum(r['static_depth_incompatible_mass'] for r in summary['records'])))
write(dest/'report.json',records);files={str(p.relative_to(dest)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in dest.rglob('*') if p.is_file()};write(dest/'manifest.json',files)
archive=ROOT/'depth_delivery_v1.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(dest,arcname='depth_delivery_v1')
write(ROOT/'depth_delivery_v1_receipt.json',dict(archive=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()));print(json.dumps(records))
