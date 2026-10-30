from pathlib import Path
import numpy as np
p=Path('work/v43_sim_mechanism_20261030/visibility_control_r1');t=np.load(p/'target_world_points.npz');v=np.load(p/'original_visibility.npz')['visible']
for f in [0,8,15]:
 u=t['anchor_frame']==f;yx=t['yx'][u];revealed=np.zeros((320,480),bool);anyv=revealed.copy();n=np.zeros((320,480),np.int16);revealed[yx[:,0],yx[:,1]]=True;n[yx[:,0],yx[:,1]]=v[:,u].sum(0);anyv=revealed&(n>0)
 np.savez_compressed(p/f'eval_masks_{f:06d}.npz',revealed_road=revealed,visible_old_any=anyv,visible_old_0=revealed&~anyv,visible_old_1=revealed&(n==1),visible_old_2plus=revealed&(n>=2),visible_old_frame_count=n)
