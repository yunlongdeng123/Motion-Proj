from pathlib import Path
import sys
from collections import Counter
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent));from temporal_factory import T,read,dump
O=T/'r10'
def main():
 cases=[c for c in read(O/'dataset_catalog.json')['cases'] if c['split']=='train'];stats={}
 for c in cases:
  folder=Path(c['folder']);hs=[np.asarray(Image.open(folder/'model_hole'/f'{i:03}.png'))>0 for i in range(10)];areas=[float(h.mean()) for h in hs]
  stats[c['dataset_id']]={'family':c['process_family'],'scene':c['receiver_scene'],'type':c['type'],'hole_canvas_fraction':float(np.mean(areas)),'border_frames':sum(h[0].any() or h[-1].any() or h[:,0].any() or h[:,-1].any() for h in hs)}
 steps=[__import__('json').loads(s) for s in (O/'training/steps.jsonl').read_text().splitlines()];counts=Counter(stats[s['last_case']]['family'] for s in steps)
 dump(O/'training_process_census.json',{'unique_case_family_counts':dict(Counter(c['process_family'] for c in cases)),'actual_step_family_counts':dict(counts),'step_weighted_hole_canvas_fraction':float(np.mean([stats[s['last_case']]['hole_canvas_fraction'] for s in steps])),'training_steps':len(steps),'source_world_counts_by_family':{k:len({c['receiver_scene'] for c in cases if c['process_family']==k}) for k in counts},'sweep_train_fraction':sum(c['process_family']=='sweep_B' for c in cases)/len(cases),'cases':stats,'limits':'frequency of actual valid process windows, not proof task distribution fully covered; one-second windows remain; physical legality and true texture correspondences not guaranteed by counts'})
 print(dict(counts))
if __name__=='__main__':main()
