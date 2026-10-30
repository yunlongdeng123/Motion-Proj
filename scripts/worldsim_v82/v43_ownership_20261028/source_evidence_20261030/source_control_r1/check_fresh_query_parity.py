from pathlib import Path
import numpy as np,json
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030");rows=[]
for f in ("000000","000008","000015"):
 a=np.load(r/"evidence_reconstruction_r1/edit_common_10k/none"/(f+".npz"))["final"];b=np.load(r/"source_control_r1/fresh_source_r1/none"/(f+".npz"))["final"];error=float(np.abs(a-b).max());assert error<=2e-6;rows.append(dict(frame=f,new_native_vs_private_source_none_max_abs=error))
p=r/"source_control_r1/fresh_comparison_delivery_r1/source/query_forward_parity.json";p.write_text(json.dumps(dict(all_passed=True,rows=rows,scope="actual same new checkpoint final FP32 native common none versus private-query source none"),indent=2));print(p.read_text())
