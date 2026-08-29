"""Predict P199 event-level Brier loss and selectively authorize low-risk estimates."""

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
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload


class AuthorityRiskHead(nn.Module):
 def __init__(self,widths):
  super().__init__();layers=[];width=11
  for h in widths:layers.extend((nn.Linear(width,int(h)),nn.SiLU()));width=int(h)
  layers.append(nn.Linear(width,1));self.network=nn.Sequential(*layers)
 def forward(self,x):return functional.softplus(self.network(x).squeeze(1))


def _select(score,coverage):
 n,b=score.shape;mask=np.zeros((n,b),bool);k=int(round(n*coverage))
 for j in range(b):mask[np.argsort(score[:,j],kind='stable')[:k],j]=True
 return mask


def _metrics(prob,truth,mask):
 brier=[];cal=[]
 for j in range(prob.shape[1]):
  take=mask[:,j];brier.append(np.mean((prob[take,j]-truth[take,j])**2));cal.append(abs(prob[take,j].mean()-truth[take,j].mean()))
 return float(np.mean(brier)),float(np.mean(cal))


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);raw=np.concatenate((scores,clearances),1);base_feature=((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32);prob=_joint_probabilities(copula,torch.from_numpy(base_feature).cuda(),torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c['evaluation']['monte_carlo_samples']),int(c['seed']));truth=np.all(costs[:,:,None]<=budgets[None,None,:],1).astype(np.float32);log_budget=np.log(budgets);log_budget=(log_budget-log_budget.mean())/log_budget.std();feature=np.concatenate((np.broadcast_to(base_feature[:,None],(len(scores),7,8)),np.broadcast_to(log_budget[None,:,None],(len(scores),7,1)),prob[:,:,None],(prob*(1-prob))[:,:,None]),2).astype(np.float32);x=torch.from_numpy(feature.reshape(-1,11)).cuda();loss_target=torch.from_numpy(((prob-truth)**2).reshape(-1).astype(np.float32)).cuda();train_rows=torch.from_numpy(np.flatnonzero(np.repeat(train,7))).cuda();m=c['model'];model=AuthorityRiskHead(m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_rows[torch.randint(len(train_rows),(int(m['batch_size']),),device='cuda')];pred=model(x[idx]);loss=functional.mse_loss(pred,loss_target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P220 authority risk step={step+1} mse={last:.6f}',flush=True)
 with torch.no_grad():risk=model(x).reshape(len(scores),7).cpu().numpy()
 coverage=float(c['evaluation']['authority_coverage']);candidate_mask=_select(risk[dev],coverage);confidence_mask=_select((prob*(1-prob))[dev],coverage);all_mask=np.ones_like(candidate_mask);cb,cc=_metrics(prob[dev],truth[dev],candidate_mask);bb,bc=_metrics(prob[dev],truth[dev],confidence_mask);ab,ac=_metrics(prob[dev],truth[dev],all_mask);checks={'learned_authority_selected_Brier_strictly_better_than_confidence_control':cb<bb,'learned_authority_selected_calibration_error_noninferior_to_confidence_control':cc<=bc};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'budget_log_mean':float(np.log(budgets).mean()),'budget_log_scale':float(np.log(budgets).std()),'coverage':coverage},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int(train.sum()),'event_row_count':int(train.sum()*7),'final_loss_prediction_mse':last},'development':{'trajectory_count':int(dev.sum()),'event_row_count':int(dev.sum()*7),'authority_coverage':coverage,'learned_authority_selected_brier':cb,'confidence_selected_brier':bb,'all_event_brier':ab,'selected_Brier_reduction_vs_confidence':(bb-cb)/bb,'learned_authority_selected_calibration_error':cc,'confidence_selected_calibration_error':bc,'all_event_calibration_error':ac,'selected_calibration_error_reduction_vs_confidence':(bc-cc)/max(bc,1e-12)},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
