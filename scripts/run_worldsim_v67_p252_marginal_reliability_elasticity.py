"""Distill P246 log-budget derivatives into a marginal reliability head."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from scipy.stats import spearmanr
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_load_density
from scripts.run_worldsim_v67_p203_monotone_beta_joint_calibration import MonotoneBetaCalibration
from scripts.run_worldsim_v67_p233_monotone_prefix_reliability_surface import _dataset
from scripts.run_worldsim_v67_p246_extended_budget_rate_spline import MonotoneRateSplineSurface


class MarginalReliabilityElasticity(nn.Module):
 def __init__(self,width):
  super().__init__();self.network=nn.Sequential(nn.Linear(37,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,4))
 def forward(self,feature,normalized_log_budget):return functional.softplus(self.network(torch.cat((feature,normalized_log_budget.reshape(-1,1)),1)))


def _teacher_elasticity(model,feature,budget_z,chunk_size):
 outputs=[]
 with torch.no_grad():
  for start in range(0,len(feature),chunk_size):
   x=torch.from_numpy(feature[start:start+chunk_size]).cuda();context=model.encoder(x);rates=functional.softplus(model.rate_knots(context).reshape(-1,4,model.knot_count));width=2./(model.knot_count-1);areas=.5*(rates[:,:,:-1]+rates[:,:,1:])*width;cumulative=torch.cat((torch.zeros_like(rates[:,:,:1]),torch.cumsum(areas,2)),2);columns=[]
   for value in budget_z:
    position=float(np.clip((float(value)+1)/width,0,model.knot_count-1));index=min(int(np.floor(position)),model.knot_count-2);fraction=position-index;r0=rates[:,:,index];r1=rates[:,:,index+1];rate=r0+(r1-r0)*fraction;integral=cumulative[:,:,index]+width*(r0*fraction+.5*(r1-r0)*fraction*fraction);units=torch.sigmoid(model.intercept(context)+integral);probability=torch.cumprod(units,1);columns.append(probability*torch.cumsum((1-units)*rate,1))
   outputs.append(torch.stack(columns,2).cpu().numpy())
 return np.concatenate(outputs)


def _evaluate(student,feature,teacher,budget_z):
 x=torch.from_numpy(feature).cuda();outputs=[];torch.cuda.synchronize();before=time.monotonic()
 with torch.no_grad():
  for value in budget_z:outputs.append(student(x,torch.full((len(x),),float(value),device='cuda')).cpu().numpy())
 torch.cuda.synchronize();forward=time.monotonic()-before;candidate=np.stack(outputs,2);correlations=[float(spearmanr(candidate[:,h,b],teacher[:,h,b]).statistic) for h in range(4) for b in range(len(budget_z))];return {'trajectory_count':int(len(feature)),'budget_count':int(len(budget_z)),'elasticity_MAE':float(np.mean(np.abs(candidate-teacher))),'mean_within_query_Spearman':float(np.mean(correlations)),'minimum_within_query_Spearman':float(np.min(correlations)),'negative_elasticity_count':int(np.sum(candidate<0)),'student_forward_seconds':forward}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p203=torch.load(a.runs_root/c['frozen_p203']['run']/c['frozen_p203']['artifact'],map_location='cuda');calibrator=MonotoneBetaCalibration().cuda();calibrator.load_state_dict(p203['model_state_dict']);calibrator.eval();h=np.asarray(c['horizons_seconds'],np.float32);anchors=np.asarray(c['feature_anchor_budgets'],np.float32);common=(members,ens,density,fd,p199,copula,calibrator,h);tail=(float(c['boundary_state_cost']['clearance_floor_m']),int(c['teacher']['monte_carlo_samples']),int(c['seed']),float(c['teacher']['ignored_future_marginal_probability']));sf,_,_,scenes,_=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common,anchors,*tail);f183,_,_,_,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common,anchors,*tail);f201,_,_,_,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common,anchors,*tail);frozen=torch.load(a.runs_root/c['frozen_p246']['run']/c['frozen_p246']['artifact'],map_location='cuda');teacher=MonotoneRateSplineSurface(int(frozen['context_width']),int(frozen['rate_knot_count'])).cuda();teacher.load_state_dict(frozen['model_state_dict']);teacher.eval();domain=np.asarray(c['training_budget_domain'],np.float32);log_min,log_max=np.log(domain);count=int(c['training_budget_count']);training_budgets=np.exp(log_min+(np.arange(count,dtype=np.float32)+float(c['training_budget_offset']))/count*(log_max-log_min));heldout_anchors=np.asarray(c['heldout_anchor_budgets'],np.float32);heldout_budgets=np.sqrt(heldout_anchors[:-1]*heldout_anchors[1:]);mean=float(frozen['budget_log_mean']);scale=float(frozen['budget_log_scale']);training_z=((np.log(training_budgets)-mean)/scale).astype(np.float32);heldout_z=((np.log(heldout_budgets)-mean)/scale).astype(np.float32);chunk=int(c['teacher_chunk_size']);source_target=_teacher_elasticity(teacher,sf,training_z,chunk);target183=_teacher_elasticity(teacher,f183,heldout_z,chunk);target201=_teacher_elasticity(teacher,f201,heldout_z,chunk);dev=scenes%int(c['split']['development_scene_modulus'])==int(c['split']['development_scene_remainder']);source_heldout_target=_teacher_elasticity(teacher,sf[dev],heldout_z,chunk);x=torch.from_numpy(sf).cuda();target=torch.from_numpy(source_target).cuda();z=torch.from_numpy(training_z).cuda();train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();m=c['student'];student=MarginalReliabilityElasticity(int(m['width'])).cuda();opt=torch.optim.AdamW(student.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];budget_idx=torch.randint(len(training_z),(len(idx),),device='cuda');prediction=student(x[idx],z[budget_idx]);truth=target[idx,:,budget_idx];loss=functional.l1_loss(prediction,truth);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P252 marginal elasticity step={step+1} mae={last:.7f}',flush=True)
 source=_evaluate(student,sf[dev],source_heldout_target,heldout_z);r183=_evaluate(student,f183,target183,heldout_z);r201=_evaluate(student,f201,target201,heldout_z);decision=c['decision'];checks={'P201_elasticity_fidelity':r201['elasticity_MAE']<=float(decision['maximum_P201_elasticity_MAE']),'P201_marginal_value_ranking':r201['mean_within_query_Spearman']>=float(decision['minimum_P201_mean_within_query_Spearman'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':student.state_dict(),'width':m['width'],'input_dimension':37,'budget_log_mean':mean,'budget_log_scale':scale,'base_model':c['frozen_p246']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'budget_count':int(len(training_z)),'final_elasticity_mae':last},'heldout_budgets':[float(v) for v in heldout_budgets],'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
