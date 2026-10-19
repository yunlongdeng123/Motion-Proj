"""CPU evaluation of the exact saved native static getters; no model updates."""
from pathlib import Path
import os,json,math,sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import torch,numpy as np
E=Path(__file__).resolve().parent;N=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018');S=N/'training_source/scene0008_089_137';R=N/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k';D=E/'gn11_static_duplicate_20261019'
sys.path.insert(0,str(S));from scene.gaussian_model import GaussianModel
def stats(v):
 v=v.detach().double().flatten();q=torch.quantile(v,torch.tensor([.5,.95,.99],dtype=torch.float64))
 return dict(mean=float(v.mean()),median=float(q[0]),p95=float(q[1]),p99=float(q[2]),max=float(v.max()))
torch.set_num_threads(2)
captured,step=torch.load(str(R/'chkpnt30000.pth'),map_location='cpu',weights_only=False,mmap=True);assert step==30000
# Bypass __init__'s CUDA-related activations; use the real class property/functions on actual CPU state.
m=object.__new__(GaussianModel);m._xyz=captured[1];m._velocity=captured[9];m._t=captured[7];m._scaling_t=captured[8];m.T=captured[16];m.velocity_decay=captured[17];m.scaling_t_activation=torch.exp
raw=m._velocity.norm(dim=1);feature=m.get_inst_velocity.norm(dim=1);damping=torch.exp(-m.get_scaling_t/m.T/2*m.velocity_decay).flatten()
scale=.1;lo=None;hi=None
for t in np.linspace(-.48,.48,49):
 xyz=m.get_xyz_SHM(float(t));lo=xyz.clone() if lo is None else torch.minimum(lo,xyz);hi=xyz.clone() if hi is None else torch.maximum(hi,xyz)
span=(hi-lo).norm(dim=1)/scale
summary=json.loads((R/'static_time_30000/summary.json').read_text());(D/'native_static_time_summary.json').write_text(json.dumps(summary,indent=2))
from PIL import Image
export=D/'fixed_camera49';export.mkdir(exist_ok=True)
for i in range(49):
 with np.load(R/'static_time_30000'/f'{i:03d}.npz',allow_pickle=False) as z:
  rgb=z['static_rgb'];assert rgb.shape==(3,320,480)
  Image.fromarray(np.rint(np.clip(rgb,0,1).transpose(1,2,0)*255).astype(np.uint8)).save(export/f'{i:03d}.png')
out=dict(status='COMPLETE_CPU_EXACT_NATIVE_STATIC_MOTION_FEATURE_AUDIT',step=step,rows=len(raw),cycle_T=float(m.T),velocity_decay=float(m.velocity_decay),
 raw_base_velocity_model=stats(raw),regularized_get_inst_velocity_model=stats(feature),damping=stats(damping),
 feature_velocity_below_1e_6_fraction=float((feature<1e-6).float().mean()),actual_sampled_center_span_m=stats(span),
 rows_center_span_over_0p1m_and_feature_velocity_below_1e_6=int(((span>.1)&(feature<1e-6)).sum()),
 native_existing_fixed_pose_time_change=dict(max_static_RGB_MAE=float(max(x['static_rgb_MAE_from_anchor'] for x in summary['rows'])),
  max_static_RGB_abs_change=float(max(x['static_rgb_max_abs_from_anchor'] for x in summary['rows'])),
  source='Already completed native30K fixed-camera49-timestamp render; no new GPU work'),
 expressions=dict(position='xyz + raw_velocity*sin((timestamp-t)*2pi/T)/(2pi/T)',regularized_feature='raw_velocity*exp(-exp(log_temporal_width)/T/2*velocity_decay)'),
 caveats=['Velocities are model-coordinate per model-time quantities, not physical m/s.',
  'Global counts do not attribute the marked car-shaped pixels to particular Gaussians.',
  'The damped-feature/raw-position formulas are inherited native behavior; this is a confirmed constraint/representation mismatch, not evidence of a newly introduced source edit.'],
 GPU_used=False,training_updates=0)
assert not torch.cuda.is_initialized();(D/'motion_feature_receipt.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
