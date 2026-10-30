import subprocess,json
from pathlib import Path
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030")
py="/root/autodl-tmp/envs/drivingeditor/bin/python"
for baseline in (True,False):
 for edit in ("none","delete","shift","rotate"):
  if not baseline and edit=="delete":continue
  out=r/"m1_assignment_r1"/("heldout_m0_"+edit if baseline else "heldout_m1_"+edit)
  args=[py,str(r/"code/reconstruction/render_heldout_m1.py"),"--selection",str(r/"m1_assignment_r1/recall_verified/selection_000008.npz"),"--event",str(r/"formal_r3_10k/checkpoint_events/iteration_010000.json"),"--output",str(out),"--reference-root",str(r/"dataset_r2/privileged"),"--edit",edit]
  if baseline:args+=["--baseline"]
  with (out.with_suffix(".log")).open("w") as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True)
