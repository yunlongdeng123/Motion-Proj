"""Read only pose/camera metadata for the completed scale diagnostic; no torch."""
from pathlib import Path
import hashlib,json,numpy as np
CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OUT=CAM/'scene008_a6_insert_artifacts_20261105/scale'
observations=CAM/'scene008_instances_20261105_r1/observations/frames'
records=[]
for raw in (106,113,130):
    p=observations/f'{raw:03d}.npz'
    with np.load(p,allow_pickle=False) as z:
        row=dict(raw=raw,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for k in ('model_from_local','model_to_camera','model_time'):
            row[k]=z[k].tolist()
    records.append(row)
(OUT/'camera_geometry_cpu.json').write_text(json.dumps(dict(records=records,torch_imported=False,GPU_calls=0),indent=2)+'\n')
print('PASS_CPU_METADATA_ONLY')
