from pathlib import Path
import json
w=Path('work/v43_sim_mechanism_20261030');new=w/'evidence_reconstruction_r1/evidence';old=Path('outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/evidence/edit_r3_10k')
for label,p in [('old',old/'reference_metrics.json'),('new',new/'edit_samequery/reference_metrics.json')]:
 j=json.loads(p.read_text());print(label)
 for r in j['rows']:
  if r['frame']=='000008':print(r['edit'],[(k,r['regions'][k].get('raw_mae'),r['regions'][k].get('clamped_mae')) for k in ['full','newly_exposed_road','original_body_footprint']])
s=json.loads((new/'observed_unseen_road_metrics.json').read_text());print('stratified pooled3anchors')
for x in s['pixel_weighted_summary']:
 if x['group'] in ['old_observed_road','old_unseen_road','all_revealed_road']:print(x['arm'],x['edit'],x['group'],x['pixels'],x['metrics'].get('raw_mae'),x['metrics'].get('clamped_mae'))
print('receiptkeys',list(json.loads((new/'completion_receipt.json').read_text())))
