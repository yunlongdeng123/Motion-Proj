"""模型调用前冻结一次更细的物理位置候选，不改验收阈值。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,read,dump
path=O/'pose_sampling_amendment.json'
if not path.exists():
    states=[read(O/f'asset_factory_state_{i}.json') for i in range(4)]
    dump(path,{'reason':'coarse two-meter lateral proposals mostly background, very few valid partial protected occlusions; old affine exact stage14cases/6scenes retained',
               'observed_data_only_states':states,'model_inference_outcomes_used':False,'new_global_proposals':[[z,x] for z in [4.5,5.5,7.,9.,12.,15.] for x in [-3.,-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.,3.]],
               'budget':'one fixed66-pose supplement/source/two retained assets; no per-case thresholds/seed/weights changes; stop this candidate pool after it completes',
               'rules_unchanged':'real LiDAR/map, .3m clearance,64px ego band, full 10-frame size/depth/order/normal-mask continuity, protected30-80/dense20-70%, all synthetic influence erased',
               'data_only_control_boundary':'r8 dataset change includes broader scenes and explicit3D silhouette proposal method; cannot attribute any later benefit solely to scene count',
               'first_shape_oracle':'silhouette_probe/results.json; optimized union equals original full-face raster at actual camera'})
    print('pose sampling registered')
