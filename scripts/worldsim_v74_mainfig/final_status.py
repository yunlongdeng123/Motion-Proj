import pathlib,json,time,shutil
from geometry_contract import R
S=R/'secondary';check={'primary_forwards':len(list((R/'predictions').glob('*/*/*/result.json'))),'secondary_forwards':len(list((S/'predictions').glob('*/*/*/result.json'))),'primary_summaries':0,'secondary_summaries':0,'native_methods':{},'gaussian_render_windows':0,'gaussian_render_views':0,'count_violations':[]}
for root,key in [(R,'primary_summaries'),(S,'secondary_summaries')]:
 for protocol in ['evaluation_bgscale','evaluation_common6']:
  for f in (root/protocol).glob('*/*/*/summary.json'):
   check[key]+=1
   for r in json.loads(f.read_text())['rows']:
    if r['status']!='EVALUATED':continue
    c=r['counts']
    assert c['hit']+c['early']+c['late']+c['miss']==c['rays']
    assert c['early_0.5']<=c['early_0.3']<=c['early']<=c['early_0.1']
    assert c['early_with_later_correct']<=min(c['early'],c['any_correct_intersection'])
for m in ['nksr','noksr']:
 f=S/m/'evaluation.json'
 if not f.exists():check['native_methods'][m]={'status':'NOT_EVALUATED'};continue
 rows=json.loads(f.read_text())['rows'];check['native_methods'][m]={'objects':len(rows),'status_counts':{k:sum(r['status']==k for r in rows) for k in set(r['status'] for r in rows)},'evaluable':sum(r.get('query_status')=='EVALUATED' for r in rows)}
 for r in rows:
  if r.get('query_status')=='EVALUATED':c=r['counts'];assert c['hit']+c['early']+c['late']+c['miss']==c['rays'];assert c['early_0.5']<=c['early_0.3']<=c['early']<=c['early_0.1']
for f in (S/'predictions/dggt').glob('*/*/native_render/result.json'):
 a=json.loads(f.read_text());check['gaussian_render_windows']+=1;check['gaussian_render_views']+=len(a['views']);assert all(r['finite_fraction']==1 for r in a['views'])
assert check['primary_forwards']==48 and check['secondary_forwards']==24 and check['primary_summaries']==72 and check['secondary_summaries']==36
assert check['gaussian_render_views']==108
check.update(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),failure_ledger_delta='V74-H2-F13',human_verdict=None,training_runs=0,independent_new_logs=0)
(R/'final_qa.json').write_text(json.dumps(check,indent=2))
old=R/'omega_status.json'
if old.exists() and not (R/'omega_status_pre_user512.json').exists():shutil.copy2(old,R/'omega_status_pre_user512.json')
old.write_text(json.dumps({'status':'DONE','selected_version':'original512 from user-provided public mirror','official_architecture_strict_load':True,'completed_forwards':12,'checkpoint':'/root/autodl-tmp/models/worldsim_v74_mainfig/vggt_omega_1b_512.pt','bytes':4576706117,'source':'https://huggingface.co/1kaiser/vggt-omega-jax/resolve/main/vggt_omega_1b_512.pt','416_reproduction_used':False,'mirror_author_identity_independently_verified':False},indent=2))
print(json.dumps(check),flush=True)
