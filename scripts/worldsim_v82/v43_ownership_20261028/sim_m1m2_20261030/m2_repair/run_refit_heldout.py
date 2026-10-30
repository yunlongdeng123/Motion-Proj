import json
from pathlib import Path
import subprocess
import sys
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030'); out=root/'m2_repair_r1'
for name in ['refit_assignment','refit_split']:
    command=[sys.executable,str(out/'code/render_recall_heldout4.py'),'--event',str(root/'formal_r3_10k/checkpoint_events/iteration_010000.json'),'--output',str(out/('heldout_'+name)),'--reference-root',str(root/'dataset_r2/privileged'),'--selection-root',str(root/'m1_assignment_r1/recall_moments'),'--policy','balanced','--repair-stage','position','--keyframes','8','--replay-geometry',str(out/name/'repaired_geometry_000008.npz')]
    with (out/('heldout_'+name+'.log')).open('w') as log:
        run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if run.returncode: raise SystemExit(run.returncode)
