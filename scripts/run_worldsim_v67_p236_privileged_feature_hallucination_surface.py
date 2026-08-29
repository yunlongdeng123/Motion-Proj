"""Hallucinate privileged P199 condition features from marginal-CDF inputs."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import MonotonePrefixSurface,_dataset,_evaluate


class FeatureHallucinator(nn.Module):
 def __init__(self,widths):
  super().__init__();layers=[];width=28
  for hidden in widths:layers.extend((nn.Linear(width,int(hidden)),nn.SiLU()));width=int(hidden)
  layers.append(nn.Linear(width,8));self.network=nn.Sequential(*layers)
 def forward(self,marginal):return self.network(marginal)


class HallucinatedSurface(nn.Module):
 def __init__(self,hallucinator,surface):super().__init__();self.hallucinator=hallucinator;self.surface=surface
 def forward(self,feature):return self.surface(torch.cat((self.hallucinator(feature[:,8:]),feature[:,8:]),1))


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h,budgets,float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,st,sy,scenes,ss=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);f183,t183,y183,_,s183=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);f201,t201,y201,_,s201=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);artifact=torch.load(a.runs_root/c['frozen_p233']['run']/c['frozen_p233']['artifact'],map_location='cuda');surface=MonotonePrefixSurface(artifact['hidden_dimensions']).cuda();surface.load_state_dict(artifact['model_state_dict']);surface.eval()
 for parameter in surface.parameters():parameter.requires_grad_(False)
 dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf[:,8:]).cuda();target=torch.from_numpy(sf[:,:8]).cuda();m=c['hallucinator'];hallucinator=FeatureHallucinator(m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(hallucinator.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];loss=functional.mse_loss(hallucinator(x[idx]),target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P236 feature hallucination step={step+1} mse={last:.7f}',flush=True)
 model=HallucinatedSurface(hallucinator,surface);source=_evaluate(model,sf,st,sy,dev);r183=_evaluate(model,f183,t183,y183);r201=_evaluate(model,f201,t201,y201);source['condition_feature_RMSE']=float(np.sqrt(np.mean((hallucinator(torch.from_numpy(sf[dev,8:]).cuda()).detach().cpu().numpy()-sf[dev,:8])**2)));r183['condition_feature_RMSE']=float(np.sqrt(np.mean((hallucinator(torch.from_numpy(f183[:,8:]).cuda()).detach().cpu().numpy()-f183[:,:8])**2)));r201['condition_feature_RMSE']=float(np.sqrt(np.mean((hallucinator(torch.from_numpy(f201[:,8:]).cuda()).detach().cpu().numpy()-f201[:,:8])**2)));decision=c['decision'];checks={'P201_surface_teacher_fidelity':r201['surface_teacher_probability_MAE']<=float(decision['maximum_P201_surface_teacher_probability_MAE']),'P201_final_curve_teacher_fidelity':r201['final_curve_teacher_probability_MAE']<=float(decision['maximum_P201_final_curve_teacher_probability_MAE']),'P201_surface_quality_noninferior':r201['relative_surface_Brier_degradation_vs_teacher']<=float(decision['maximum_relative_P201_surface_Brier_degradation']) and r201['absolute_surface_calibration_error_increase_vs_teacher']<=float(decision['maximum_absolute_P201_surface_calibration_error_increase'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':hallucinator.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'input_dimension':28,'output_dimension':8,'frozen_surface':c['frozen_p233']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_privileged_feature_mse':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'teacher_MC_seconds':{'source':ss,'P183':s183,'P201':s201},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
