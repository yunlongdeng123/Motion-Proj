"""Pairwise proper-loss ranking for selective P199 authority."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p220_selective_joint_reliability_authority import AuthorityRiskHead,_metrics,_select


def _dataset(path,members,ens,density,fd,p199,copula,h,budgets,floor,samples,seed):
 with np.load(path,allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,floor),h);norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);raw=np.concatenate((scores,clearances),1);base=((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32);prob=_joint_probabilities(copula,torch.from_numpy(base).cuda(),torch.from_numpy(marginal.astype(np.float32)).cuda(),samples,seed);truth=np.all(costs[:,:,None]<=budgets[None,None,:],1).astype(np.float32);lb=(np.log(budgets)-np.log(budgets).mean())/np.log(budgets).std();feature=np.concatenate((np.broadcast_to(base[:,None],(len(scores),7,8)),np.broadcast_to(lb[None,:,None],(len(scores),7,1)),prob[:,:,None],(prob*(1-prob))[:,:,None]),2).astype(np.float32);return feature,prob,truth,scenes


def _evaluate(model,feature,prob,truth,coverage,index=None):
 if index is not None:feature=feature[index];prob=prob[index];truth=truth[index]
 with torch.no_grad():risk=model(torch.from_numpy(feature.reshape(-1,11)).cuda()).reshape(len(feature),7).cpu().numpy()
 candidate=_select(risk,coverage);control=_select(prob*(1-prob),coverage);all_mask=np.ones_like(candidate);cb,cc=_metrics(prob,truth,candidate);bb,bc=_metrics(prob,truth,control);ab,ac=_metrics(prob,truth,all_mask);return {'trajectory_count':int(len(feature)),'event_row_count':int(len(feature)*7),'learned_authority_selected_brier':cb,'confidence_selected_brier':bb,'all_event_brier':ab,'selected_Brier_reduction_vs_confidence':(bb-cb)/bb,'learned_authority_selected_calibration_error':cc,'confidence_selected_calibration_error':bc,'all_event_calibration_error':ac,'selected_calibration_error_reduction_vs_confidence':(bc-cc)/max(bc,1e-12)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();h=np.asarray(c['horizons_seconds'],np.float32);budgets=np.asarray(c['reliability_budgets'],np.float32);floor=float(c['boundary_state_cost']['clearance_floor_m']);samples=int(c['evaluation']['monte_carlo_samples']);seed=int(c['seed']);common=(members,ens,density,fd,p199,copula,h,budgets,floor,samples,seed);sf,sp,st,scenes=_dataset(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],*common);f183,p183,t183,_=_dataset(a.runs_root/c['p183_rows']['run']/c['p183_rows']['artifact'],*common);f201,p201,t201,_=_dataset(a.runs_root/c['p201_rows']['run']/c['p201_rows']['artifact'],*common);split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train_idx=torch.from_numpy(np.flatnonzero(~dev)).cuda();x=torch.from_numpy(sf).cuda();realized=torch.from_numpy((sp-st)**2).cuda();m=c['model'];model=AuthorityRiskHead(m['hidden_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.;batch=int(m['pair_batch_size'])
 for step in range(int(m['steps'])):
  i=train_idx[torch.randint(len(train_idx),(batch,),device='cuda')];j=train_idx[torch.randint(len(train_idx),(batch,),device='cuda')];b=torch.randint(7,(batch,),device='cuda');li=realized[i,b];lj=realized[j,b];direction=torch.sign(li-lj);difference=model(x[i,b])-model(x[j,b]);loss=functional.softplus(-direction*difference).mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P223 pairwise authority step={step+1} logistic={last:.6f}',flush=True)
 coverage=float(c['evaluation']['authority_coverage']);source=_evaluate(model,sf,sp,st,coverage,dev);r183=_evaluate(model,f183,p183,t183,coverage);r201=_evaluate(model,f201,p201,t201,coverage);checks={'P201_pairwise_authority_selected_Brier_strictly_better_than_confidence_control':r201['selected_Brier_reduction_vs_confidence']>0,'P201_pairwise_authority_selected_calibration_error_noninferior_to_confidence_control':r201['selected_calibration_error_reduction_vs_confidence']>=0};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'hidden_dimensions':m['hidden_dimensions'],'budget_log_mean':float(np.log(budgets).mean()),'budget_log_scale':float(np.log(budgets).std()),'coverage':coverage},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int((~dev).sum()),'event_row_count':int((~dev).sum()*7),'final_pairwise_logistic_loss':last},'source_development':source,'P183_consumed_development':r183,'P201_post_hoc_recovery_development':r201,'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
