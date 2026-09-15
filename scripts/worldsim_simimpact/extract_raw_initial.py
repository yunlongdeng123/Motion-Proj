import os
"""Extract a small, explicit camera/LiDAR list from a locally mounted public shard."""
import json,subprocess
from pathlib import Path
R=Path(os.environ.get('SIMIMPACT_RUN_ROOT','/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1'))
d=json.loads((R/'raw_source_availability.json').read_text())
out=R/'raw_initial';out.mkdir(exist_ok=True)
targets=sorted({x['filename'] for s in d.values() for x in s['records'] if ('/CAM_' in x['filename'] and x['sample_token'] in s['samples'][:2]) or '/LIDAR_TOP/' in x['filename']})
listing=R/'raw_initial_members.txt';listing.write_text('\n'.join(targets)+'\n')
archive='/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz'
# The first two log prefixes are mapped to shard01 in existing member indices;
# the adjacent third log is a discovery probe, not assumed present.
command=['tar','-xzf',archive,'-C',str(out),'--no-recursion','-T',str(listing)]
print('EXTRACT',len(targets),'members',flush=True)
p=subprocess.run(command)
results={'archive':archive,'tar_returncode':p.returncode,'requested':len(targets),
         'present':[s for s in targets if (out/s).is_file()],
         'missing':[s for s in targets if not (out/s).is_file()]}
(R/'raw_extraction_result.json').write_text(json.dumps(results,indent=2))
print({k:v if not isinstance(v,list) else len(v) for k,v in results.items()},flush=True)
