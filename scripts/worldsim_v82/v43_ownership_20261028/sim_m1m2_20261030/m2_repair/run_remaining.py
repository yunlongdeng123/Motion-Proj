import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
out=root/'m2_repair_r1'
receipts=[]
for policy,stage,name in [('strict','position_scale','strict_position_scale'),('balanced','position','balanced_position'),('balanced','position_scale','balanced_position_scale')]:
    command=[sys.executable,str(out/'code/render_m2.py'),'--event',str(root/'formal_r3_10k/checkpoint_events/iteration_010000.json'),'--output',str(out/name),'--reference-root',str(root/'dataset_r2/privileged'),'--policy',policy,'--repair-stage',stage]
    started=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipts.append(dict(name=name,returncode=run.returncode,wall_seconds=time.monotonic()-started,command=command))
    (out/'queue_receipt.json').write_text(json.dumps(receipts,indent=2))
    if run.returncode:
        raise SystemExit(run.returncode)
print(json.dumps(receipts))
