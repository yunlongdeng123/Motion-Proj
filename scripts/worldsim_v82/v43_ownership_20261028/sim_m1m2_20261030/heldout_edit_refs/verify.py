import json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
o=r/'heldout_edit_refs_r1'
a=np.asarray(Image.open(o/'none/000008.png').convert('RGB')).astype(int)
b=np.asarray(Image.open(r/'dataset_r2/privileged/heldout/000008.png').convert('RGB')).astype(int)
m=json.loads((o/'manifest.json').read_text())
m['none_parity']={'max_abs_rgb_8bit':int(abs(a-b).max()),'mean_abs_rgb_8bit':float(abs(a-b).mean()),'nonzero_channels':int(np.count_nonzero(a-b)),'original':'dataset_r2/privileged/heldout/000008.png'}
m['status']='complete' if m['none_parity']['max_abs_rgb_8bit']==0 else 'parity_difference_requires_review'
m['files_sha256']={str(p.relative_to(o)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(o.glob('*/*.png'))}
(o/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'status':m['status'],'none_parity':m['none_parity'],'sha':m['files_sha256']}))
