"""CPU: trace current OBB-center outliers to initial geometry, not just labels."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import torch
torch.set_num_threads(2)
from identity_common import *

def main():
    seed=dict(np.load(R15/'data/seed.npz'));ck=R15/'paused_checkpoint_004000_20261017/checkpoint.pt'
    c=torch.load(ck,map_location='cpu',weights_only=False)
    a={k[6:]:v.numpy() for k,v in c['state'].items() if k.startswith('field.')}
    half=np.asarray(read(c['config']['box_half_sizes_path'])['half_sizes_m']);records=[]
    for actor in ACTORS:
        rows=np.flatnonzero(a['owner']==actor);p=a['sgcm_lineage'][rows];linked=(p>=0)&(p<len(seed['owner']))
        inside=np.zeros(len(rows),bool);inside[linked]=(np.abs(seed['xyz'][p[linked]])<=half[actor]).all(-1)
        nowout=(np.abs(a['xyz'][rows])>half[actor]).any(-1)
        assert np.all(seed['owner'][p[linked]]==actor)
        records.append(dict(actor=actor,current_count=len(rows),current_outside=int(nowout.sum()),
            outside_from_initial_inside_parent=int((nowout&linked&inside).sum()),
            outside_from_initial_outside_parent=int((nowout&linked&~inside).sum()),
            outside_from_new_pool_lineage=int((nowout&~linked).sum()),
            unique_initial_inside_parents_with_outside_descendants=len(np.unique(p[nowout&linked&inside])),
            unique_initial_outside_parents_with_outside_descendants=len(np.unique(p[nowout&linked&~inside])),
            current_inside_from_initial_outside_parent=int((~nowout&linked&~inside).sum())))
    write(TASK/'tristate/boundary_origins.json',dict(status='PROVENANCE_MATCHED_GEOMETRY_DIAGNOSTIC',records=records,
        definition='Gaussian center outside exact actor half-box in native actor coordinates; no opacity filter',
        limitations=['A 3D box is a diagnostic boundary, not ground-truth geometry or ownership.',
            'Initial inside parent does not prove a valid vehicle measurement; pool descendants are separate.',
            'Counts are descendants; topology and position optimization both contribute.']))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
