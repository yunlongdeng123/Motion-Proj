from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent.parent
query=json.loads((ROOT/'cli_keep10k/query.json').read_text())
assert query['status']=='PASS_READ_ONLY_NATIVE_QUERY' and query['dynamic_actor_ids']==[5,6,18,26,37]
assert query['delete_actor_ids']==[] and query['raw_ids']==[97,105,109] and query['optimization_updates']==0
assert not query['reads_original_PLY_or_masks'] and query['parameters_unchanged']
rows=[]
for raw in [97,105,109]:
    a=ROOT/f'cli_keep10k/{raw:03d}.png';b=ROOT/f'formal10k_keep_pilot/frames/{raw:03d}/keep.png'
    left=np.asarray(Image.open(a).convert('RGB'));right=np.asarray(Image.open(b).convert('RGB'))
    assert left.shape==right.shape==(320,480,3)
    delta=np.abs(left.astype(np.int16)-right.astype(np.int16));assert int(delta.max())<=1
    rows.append(dict(raw=raw,max_uint8_difference=int(delta.max()),mean_uint8_difference=float(delta.mean()),
        cli_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),native_keep_sha256=hashlib.sha256(b.read_bytes()).hexdigest()))
result=dict(status='PASS_PUBLIC_CLI_KEEP_MATCHES_ACCEPTED_NATIVE_KEEP',model=str(ROOT/'formal10k'),iteration=10000,
    frames=rows,raw_ids=[97,105,109],optimization_updates=0,GPU_used_by_this_check=False,
    public_loader_requires_GT_PLY_masks=False,delete_actor_ids=[],quality_verdict_not_inferred=True,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'public_keep_png_consistency.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
