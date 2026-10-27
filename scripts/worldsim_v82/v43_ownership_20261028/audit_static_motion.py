"""CPU: original static batch actually has learned motion and temporal opacity."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import numpy as np
import torch
R=Path('/root/autodl-tmp/drivingeditor_repro/campaign');CTX='14705303724557273004_3105_000_3125_000';D=R/'v43_ownership_20261028/target_unknown_v2'

def main():
    out=D/'static_motion_audit';out.mkdir(exist_ok=False);run=R/'persistent_world_20261027/native_scene_training_v1'/CTX/'formal30k';a,step=torch.load(str(run/'chkpnt30000.pth'),map_location='cpu',mmap=True);T=np.loadtxt(R/'persistent_world_20261027/motion_scene_inputs_v1'/CTX/'transform_matrix.txt');scale=np.linalg.norm(T[:3,0]);t=a[7].detach().numpy().ravel();sigma=a[8].detach().exp().numpy().ravel();velocity=a[9].detach().numpy();period=float(a[16]);angular=2*np.pi/period;times=np.arange(49)*.02-.48;count=np.zeros(len(t),np.int16);low=np.full(len(t),np.inf);high=np.full(len(t),-np.inf)
    for time in times:
        live=np.exp(-.5*((time-t)/sigma)**2)>.05;factor=np.sin((time-t)*angular)/angular;low=np.minimum(low,np.where(live,factor,np.inf));high=np.maximum(high,np.where(live,factor,-np.inf));count+=live
    travel=np.linalg.norm(velocity,axis=1)/scale*np.where(count>=2,high-low,0);ev=np.load(D/'evidence_probe/evidence.npz');masks=dict(all=np.ones(len(t),bool),target_candidate=ev['target_candidate'],background_protected=ev['background_protected'],mixed=ev['mixed_target_background']);report={}
    for name,m in masks.items():report[name]=dict(rows=int(m.sum()),visible_trajectory_span_m_quantiles=np.quantile(travel[m],[0,.5,.9,.99,1]).tolist(),moved_gt10cm_rows=int((m&(travel>.1)).sum()),moved_gt50cm_rows=int((m&(travel>.5)).sum()),visible_fewer_than40_of49_rows=int((m&(count<40)).sum()))
    stable=(travel<=.1)&(count>=40);np.savez_compressed(out/'rows.npz',visible_travel_m=travel.astype('float32'),visible_frames=count,stable_candidate=stable)
    result=dict(context=CTX,original_static_rows=len(t),model_to_world_scale=scale,motion_period_model_units=period,model_time_span=[float(times[0]),float(times[-1])],physical_time_span_seconds=4.8,motion_formula='xyz(t)=xyz0+velocity*sin((t-center_time)*2pi/T)/(2pi/T); opacity(t)=opacity0*exp(-0.5*((t-center_time)/sigma_t)^2)',branch_is_not_motion_state=True,statistics=report,source_checkpoint_unchanged=True,training_updates=0)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
