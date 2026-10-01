from pathlib import Path
import sys
from collections import Counter
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,read,dump
O=T/'r10'
def main():
 cases=[c for c in read(O/'dataset_catalog.json')['cases'] if c['split']=='train'];stats={}
 old_process={r['case_id']:r for r in read(T/'r9/temporal_audit.json')['suites']['r8']['rows'] if r['split']=='train'}
 for c in cases:
  folder=Path(c['folder']);hs=[np.asarray(Image.open(folder/'model_hole'/f'{i:03}.png'))>0 for i in range(10)];areas=[float(h.mean()) for h in hs]
  proc=c.get('process') or old_process[c['case_id']];protected=[]
  for i,h in enumerate(hs):
   ms=[np.asarray(Image.open(p))>0 for p in sorted((folder/'protected').glob(f'{i:03}*.png'))];b=np.logical_or.reduce(ms) if ms else np.zeros_like(h);protected.append(float((b&h).mean()))
  stats[c['dataset_id']]={'family':c['process_family'],'scene':c['receiver_scene'],'type':c['type'],'hole_canvas_fraction':float(np.mean(areas)),'protected_hidden_canvas_fraction':float(np.mean(protected)),'border_frames':int(sum(h[0].any() or h[-1].any() or h[:,0].any() or h[:,-1].any() for h in hs)), 'actual_flags':{k:proc[k] for k in ['static_A_moving_ego_image_change','sweep_over_any_B','visibility_transition_any_B']}}
 steps=[__import__('json').loads(s) for s in (O/'training/steps.jsonl').read_text().splitlines()];counts=Counter(stats[s['last_case']]['family'] for s in steps)
 flags=['static_A_moving_ego_image_change','sweep_over_any_B','visibility_transition_any_B']
 dump(O/'training_process_census.json',{'unique_case_family_counts':dict(Counter(c['process_family'] for c in cases)),'actual_step_family_counts':dict(counts),'actual_case_process_flags':{k:sum(r['actual_flags'][k] for r in stats.values()) for k in flags},'actual_step_process_flags':{k:sum(stats[s['last_case']]['actual_flags'][k] for s in steps) for k in flags},'sampled_case_counts':dict(Counter(s['last_case'] for s in steps)),'step_weighted_hole_canvas_fraction':float(np.mean([stats[s['last_case']]['hole_canvas_fraction'] for s in steps])),'step_weighted_protected_hidden_canvas_fraction':float(np.mean([stats[s['last_case']]['protected_hidden_canvas_fraction'] for s in steps])),'training_steps':len(steps),'source_world_counts_by_family':{k:len({c['receiver_scene'] for c in cases if c['process_family']==k}) for k in counts},'sweep_train_fraction':sum(c['process_family']=='sweep_B' for c in cases)/len(cases),'cases':stats,'limits':'family labels are selection strata; actual flags overlap and include retained controls. Counts are sampling frequency, not gradient contribution. Native pixel share not latent gradient share (VAE mixes space); one-second windows, physical legality and true texture correspondences not guaranteed by counts'})
 print(dict(counts))
if __name__=='__main__':main()
