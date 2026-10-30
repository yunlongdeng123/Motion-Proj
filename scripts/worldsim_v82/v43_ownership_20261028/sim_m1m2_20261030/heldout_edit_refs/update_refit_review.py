import json
from pathlib import Path
w=Path('work/v43_sim_mechanism_20261030');b=w/'m2_repair/evidence';p=w/'heldout_edit_refs/visual_review.json';r=json.loads(p.read_text(encoding='utf-8'));out=[]
for arm in ['refit_assignment','refit_split']:
 for held in [False,True]:
  a=b/(('heldout_' if held else '')+arm)
  data=json.loads((a/('heldout_edit_metrics.json' if held else 'reference_metrics.json')).read_text())
  rows=data['rows'] if isinstance(data,dict) else data
  for x in rows:out.append({'arm':arm,'heldout':held,'edit':x['edit'],**{k:v for k,v in x['regions']['full'].items() if 'mae' in k}})
r['refit_results_reviewed']=True
r['refit_code_review']['assessment']='Executed equal-step original-RGB appearance control verified from final_completion_receipt.json and actual train/heldout four-edit images. No pending refit review.'
r['refit_code_review']['save_restore']='Actual completion receipt records per-arm restore render max difference 0.0.'
r['refit_final_receipt']=json.loads((b/'final_completion_receipt.json').read_text())['local_refit']
r['refit_visual_findings']=['M2 refit reduces none-view purple color shift and some colored boundary noise relative to frozen M2; original red vehicle appearance is partly recovered.','Compared with M1 refit, M2 refit still has additional colored edge fragments in none, including heldout; it does not visibly exceed the assignment baseline in original-view fidelity.','Delete still contains a large pink/cyan car silhouette at original car position in both refit arms; M2 has fewer peripheral speckles in places but no continuous road restoration.','Shift still shows displaced red car plus original-position pink/cyan residual in both arms.','Rotate remains a saturated green/red fragmented smear for both; M2 reduces the long bright leftward streak seen in M1 but does not recover a coherent rotated car.','Heldout reproduces these limited appearance improvements and major edit failures; no clean novel-camera editing result.']
r['refit_interpretation']='Negative result for this specific responsibility-plus-first-hit-moment repair, even after matched 200-update DC/opacity adaptation. Limited none appearance and peripheral-artifact improvements are real but do not resolve edit failure. Equal steps are not equal capacity (50712 vs 92540 parameters). This does not establish geometry repair is useless.'
r['refit_metrics']=out
r['next_research_priority']='Audit the car-shaped residual contributions that survive high-recall routing against independent surface/depth support, before adding more repair capacity: missing responsibility coverage is a direct unresolved bottleneck that local geometry/appearance operations on selected rows cannot remove.'
r['contact_sheets']+=['refit_train_contact.png','refit_heldout_contact.png'];r['human_verdict']=None
p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
