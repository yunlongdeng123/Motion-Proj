"""Remove an unused alpha placeholder; never treat it as a measurement."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parent.parent
p=RUN/'data/sealed_targets.npz';meta=RUN/'data/collection.json'
shutil.copyfile(p,p.with_name('sealed_targets_before_unused_field_cleanup.npz'))
shutil.copyfile(meta,meta.with_name('collection_before_unused_field_cleanup.json'))
x=dict(np.load(p));x.pop('operator_alpha')
with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**x)
p.with_suffix('.tmp').replace(p)
m=json.loads(meta.read_text());m.pop('alpha_note');m['operator_type']='Normalized isolated RGB only; no alpha statistics are inferred from this operator.'
m['target_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();meta.write_text(json.dumps(m,indent=2)+'\n')
