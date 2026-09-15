import pathlib,json,numpy as np
from geometry_contract import R
from evaluate_surfaces import measure
V=np.array([[-1,-1,2],[1,-1,2],[0,1,2],[-1,-1,2.35],[1,-1,2.35],[0,1,2.35]],dtype='float32');F=np.array([[0,1,2],[3,4,5]],dtype='uint32');O=np.array([[0,0,0]],dtype='float32');d=np.array([[0,0,1]],dtype='float32');r=np.array([2.35],dtype='float32')
c,t,ids,dist,anygood=measure(V,F,O,d,r);assert abs(t[0]-2)<1e-6 and ids[0]==0 and c['early']==1 and c['early_with_later_correct']==1
results={'analytical_two_surfaces':{'expected_first':2.0,'observed_first':float(t[0]),'expected_early_gap':.35,'observed_early_gap':float(r[0]-t[0]),'later_correct_found':bool(anygood[0])},'production_counts_partition':[]}
for f in (R/'evaluation_bgscale').glob('*/*/*/summary.json'):
 s=json.loads(f.read_text());bad=[]
 for row in s['rows']:
  if row['status']!='EVALUATED':continue
  n=row['counts'];assert n['rays']==n['early']+n['hit']+n['late']+n['miss'];assert n['early_with_later_correct']<=n['early'];assert n['early_0.5']<=n['early_0.3']<=n['early']<=n['early_0.1']
 results['production_counts_partition'].append(str(f.relative_to(R)))
results['scale_mask_correction']='Before final evaluation, source inspection showed diagnostic_mask is a random 1/5 subset, not a background mask. Corrected to diagnostic_mask & empty owner, as originally registered. Prior partial evaluation directory retained but excluded. No new model inference needed.'
(R/'readout_qa.json').write_text(json.dumps(results,indent=2));print(json.dumps({'analytical':results['analytical_two_surfaces'],'validated_windows':len(results['production_counts_partition'])}))
