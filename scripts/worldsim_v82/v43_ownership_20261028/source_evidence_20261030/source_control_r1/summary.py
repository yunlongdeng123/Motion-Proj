from pathlib import Path
import json
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/source_control_r1/run_r1")
for p in sorted(r.glob("source_*/source_receipt.json")):
 x=json.loads(p.read_text());print(x["frame"],"original retained",x["retained_static_original_body_alphaT"],"post",x["postdelete_body_alphaT"],"newly",x["postdelete_newly_revealed_body_alphaT"],"mixed",x["postdelete_mixed_tail_body_alphaT"])
 for a in x["cases"]:print(a["case"],a["regions"]["original_body_revealed_road"]["raw_mae"],a["regions"]["original_body_revealed_road"]["clamped_mae"],a["regions"]["road_outside_original_body"]["clamped_mae"])
