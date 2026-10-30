import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
out=root/'m2_repair_r1'; receipts=[]
for stage in ['position','position_scale','position_scale_split']:
    name='recall_'+stage+'_depth_r2'
    base=[sys.executable,str(out/'code/render_m2_recall.py'),'--event',str(root/'formal_r3_10k/checkpoint_events/iteration_010000.json'),'--output',str(out/name),'--reference-root',str(root/'dataset_r2/privileged'),'--selection-root',str(root/'m1_assignment_r1/recall_moments'),'--policy','balanced','--repair-stage',stage]
    started=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        run=subprocess.run(base,stdout=log,stderr=subprocess.STDOUT)
    receipts.append(dict(name=name,returncode=run.returncode,wall_seconds=time.monotonic()-started))
    (out/'depth_replay_queue_receipt.json').write_text(json.dumps(receipts,indent=2))
    if run.returncode: raise SystemExit(run.returncode)
    heldout=[sys.executable,str(out/'code/render_recall_heldout4.py'),'--event',str(root/'formal_r3_10k/checkpoint_events/iteration_010000.json'),'--output',str(out/('heldout_'+name)),'--reference-root',str(root/'dataset_r2/privileged'),'--selection-root',str(root/'m1_assignment_r1/recall_moments'),'--policy','balanced','--repair-stage',stage,'--keyframes','8','--replay-geometry',str(out/name/'repaired_geometry_000008.npz')]
    started=time.monotonic()
    with (out/('heldout_'+name+'.log')).open('w') as log:
        run=subprocess.run(heldout,stdout=log,stderr=subprocess.STDOUT)
    receipts.append(dict(name='heldout_'+name,returncode=run.returncode,wall_seconds=time.monotonic()-started))
    (out/'depth_replay_queue_receipt.json').write_text(json.dumps(receipts,indent=2))
    if run.returncode: raise SystemExit(run.returncode)
print(json.dumps(receipts))
