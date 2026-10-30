import numpy as np
from pathlib import Path
p=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/dataset_r2')
for f in [p/'privileged/original/000008.npz',p/'privileged/references/delete/geometry/000008.npz',p/'observations/lidar/000008.npz',p/'privileged/geometry/mesh_reference.npz']:
 print(str(f.relative_to(p)),[(k,a.shape,str(a.dtype)) for k,a in np.load(f).items()])
