"""Authorize an entire seven-budget trajectory reliability curve at once."""

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
from scripts.run_worldsim_v67_p223_pairwise_selective_authority_recovery import _dataset


class CurveAuthority(nn.Module):
 def __init__(self,widths):
  super().__init__();layers=[];width=22
  for h in widths:layers.extend((nn.Linear(width,int(h)),nn.SiLU()));width=int(h)
  layers.append(nn.Linear(width,1));self.network=nn.Sequential(*layers)
 def forward(self,x):return functional.softplus(self.network(x).squeeze(1))


def _curve_feature(event_feature,prob):return np.concatenate((event_feature[:,0,:8],prob,prob*(1-prob)),1).astype(np.float32)


def _evaluate(model,feature,prob,truth,coverage,index=None):
 if index is not None:feature=feature[index];prob=prob[index];truth=truth[index]
 with torch.no_grad():risk=model(torch.from_numpy(feature).cuda()).cpu().numpy()
 k=int(round(len(risk)*coverage));candidate=np.argsort(risk,kind='stable')[:k];control=np.argsort(np.mean(prob*(1-prob),1),kind='stable')[:k]
 def metrics(take):return float(np.mean((prob[take]-truth[take])**2)),float(np.mean(np.abs(prob[take].mean(0)-truth[take].mean(0))))
 cb,cc=metrics(candidate);bb,bc=metrics(control);ab,ac=metrics(np.arange(len(prob)));return {'trajectory_count':int(len(prob)),'authority_coverage':coverage,'learned_authority_selected_integrated_brier':cb,'confidence_selected_integrated_brier':bb,'all_trajectory_integrated_brier':ab,'selected_Brier_reduction_vs_confidence':(bb-cb)/bb,'learned_authority_selected_calibration_error':cc,'confidence_selected_calibration_error':bc,'all_trajectory_calibration_error':ac,'selected_calibration_error_reduction_vs_confidence':(bc-cc)/max(bc,1e-12)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);floor=float(c['boundary_state_cost']['clearance_floor_m']);samples=int(c['evaluation']['monte_carlo_samples']);seed=int(c['seed']);common=(members,ens,density,fd,p199,copula,h,budgets,floor,samples,seed);sef,sp,st,scenes=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);e183,p183,t183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);e201,p201,t201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);sf=_curve_feature(sef,sp);f183=_curve_feature(e183,p183);f201=_curve_feature(e201,p201);split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();target=torch.from_numpy(np.mean((sp-st)**2,1).astype(np.float32)).cuda();m=c['model'];model=CurveAuthority(m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];loss=functional.mse_loss(model(x[idx]),target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P224 trajectory curve authority step={step+1} mse={last:.6f}',flush=True)
 coverage=float(c['evaluation']['authority_coverage']);source=_evaluate(model,sf,sp,st,coverage,dev);r183=_evaluate(model,f183,p183,t183,coverage);r201=_evaluate(model,f201,p201,t201,coverage);checks={'P201_curve_authority_selected_Brier_strictly_better_than_confidence_control':r201['selected_Brier_reduction_vs_confidence']>0,'P201_curve_authority_selected_calibration_error_noninferior_to_confidence_control':r201['selected_calibration_error_reduction_vs_confidence']>=0};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'coverage':coverage},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'final_integrated_loss_prediction_mse':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
