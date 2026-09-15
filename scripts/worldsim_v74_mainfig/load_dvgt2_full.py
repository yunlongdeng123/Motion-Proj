"""Instantiate the official full DVGT2 graph; restore required anchor initialization files from checkpoint buffers."""
import pathlib,torch,numpy as np
def build(checkpoint):
 from dvgt.models.architectures.dvgt2 import DVGT2
 state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);state=state.get('model',state)
 out=pathlib.Path(checkpoint).parent/'dvgt2_checkpoint_anchors';out.mkdir(exist_ok=True)
 prefix='ego_pose_head.traj_pose_decoder.'
 for name,norm,lo,hi in [('future','norm_traj_anchor','traj_min','traj_max'),('past','norm_pose_translation_anchor','pose_translation_min','pose_translation_max')]:
  value=((state[prefix+norm]+1)/2*(state[prefix+hi]-state[prefix+lo]+1e-6)+state[prefix+lo])[0]/.1
  np.savez(out/(name+'.npz'),centers=value.numpy())
 conf={'_target_':'dvgt.models.heads.dvgt2_ego_pose_head.DVGT2EgoPoseHead','future_frame_window':8,'relative_pose_window':1,'traj_anchor_filepath':str(out/'future.npz'),'pose_translation_anchor_filepath':str(out/'past.npz'),'gt_scale_factor':.1,'enable_ego_status':False}
 # Full strict loading then overwrites these initialization buffers with the checkpoint values exactly.
 return DVGT2(dino_v3_weight_path=None,frames_chunk_size=1,ego_pose_head_conf=conf)
