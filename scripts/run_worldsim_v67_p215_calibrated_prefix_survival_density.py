"""Train prefix-cost density and a monotone beta layer on disjoint source scenes."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _mixture_nll,_predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p214_prefix_survival_max_cost_density import PrefixSurvivalDensity,_cdf,_prefix_copula_probabilities


def _all_prefix_cdf(model,tokens,masks,budgets,index):
 x=tokens[torch.from_numpy(np.flatnonzero(index)).cuda()]
 flat=x[:,None].expand(-1,4,-1,-1).reshape(-1,4,3)
 flat_mask=masks[None].expand(len(x),-1,-1).reshape(-1,4)
 return _cdf(model,flat,flat_mask,budgets).reshape(len(x),4,-1)


def _beta(prob,raw_a,raw_b,bias):
 p=prob.clamp(1e-5,1-1e-5);a=functional.softplus(raw_a);b=functional.softplus(raw_b)
 return torch.sigmoid(a*torch.log(p)-b*torch.log1p(-p)+bias[None,:,None])


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True
 ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 marginal_density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);split=c['split'];rem=scenes%int(split['scene_modulus']);dev=rem==int(split['development_remainder']);cal=rem==int(split['calibration_remainder']);train=~(dev|cal);raw_tokens=np.stack((scores,clearances,np.broadcast_to(h[None],scores.shape)),2);mean=raw_tokens[train].reshape(-1,3).mean(0);scale=raw_tokens[train].reshape(-1,3).std(0).clip(1e-6);tokens=torch.from_numpy(((raw_tokens-mean)/scale).astype(np.float32)).cuda();prefix_cost=np.maximum.accumulate(costs,1);target=torch.from_numpy(np.log1p(prefix_cost).astype(np.float32)).cuda();prefix_masks=torch.tril(torch.ones((4,4),dtype=torch.bool,device='cuda'));train_idx=torch.from_numpy(np.flatnonzero(train)).cuda();m=c['density_model'];model=PrefixSurvivalDensity(int(m['component_count']),int(m['token_dimension']),m['head_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  trajectory=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];prefix=torch.randint(4,(len(trajectory),),device='cuda');logits,means,scales=model(tokens[trajectory],prefix_masks[prefix]);loss=_mixture_nll(logits,means,scales,target[trajectory,prefix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P215 density step={step+1} nll={last:.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);cal_raw=torch.from_numpy(_all_prefix_cdf(model.eval(),tokens,prefix_masks,budgets,cal)).cuda();cal_truth=torch.from_numpy((prefix_cost[cal,:,None]<=budgets[None,None,:]).astype(np.float32)).cuda();raw_a=torch.nn.Parameter(torch.tensor(0.,device='cuda'));raw_b=torch.nn.Parameter(torch.tensor(0.,device='cuda'));bias=torch.nn.Parameter(torch.zeros(4,device='cuda'));co=torch.optim.Adam((raw_a,raw_b,bias),lr=float(c['calibrator']['learning_rate']));clast=0.
 for step in range(int(c['calibrator']['steps'])):
  out=_beta(cal_raw,raw_a,raw_b,bias);loss=functional.binary_cross_entropy(out,cal_truth);co.zero_grad(set_to_none=True);loss.backward();co.step();clast=float(loss.detach())
  if step%300==0:print(f'P215 beta step={step+1} logloss={clast:.6f}',flush=True)
 dev_raw=torch.from_numpy(_all_prefix_cdf(model.eval(),tokens,prefix_masks,budgets,dev)).cuda();candidate=_beta(dev_raw,raw_a,raw_b,bias).detach().cpu().numpy();norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(marginal_density,scores[dev,i],np.full(dev.sum(),h[i],np.float32),clearances[dev,i],budgets,norms) for i in range(4)],1);raw=np.concatenate((scores,clearances),1);p199_features=torch.from_numpy(((raw[dev]-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();baseline=_prefix_copula_probabilities(copula,p199_features,torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c['evaluation']['monte_carlo_samples']),int(c['seed']));truth=prefix_cost[dev,:,None]<=budgets[None,None,:];cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((baseline-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(baseline.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'macro_prefix_integrated_Brier_strictly_better_than_P199':cb<pb,'macro_prefix_calibration_error_noninferior_to_P199':cr>=float(c['decision']['minimum_macro_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];artifact={'model_state_dict':model.state_dict(),'token_mean':mean,'token_scale':scale,'component_count':m['component_count'],'token_dimension':m['token_dimension'],'head_dimensions':m['head_dimensions'],'horizons_seconds':h,'beta_raw_a':raw_a.detach().cpu(),'beta_raw_b':raw_b.detach().cpu(),'beta_prefix_bias':bias.detach().cpu()};torch.save(artifact,d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'density_trajectory_count':int(train.sum()),'calibration_trajectory_count':int(cal.sum()),'development_trajectory_count':int(dev.sum()),'final_density_nll':last,'final_calibration_logloss':clast,'beta_a':float(functional.softplus(raw_a)),'beta_b':float(functional.softplus(raw_b)),'beta_prefix_bias':bias.detach().cpu().tolist()},'development':{'macro_prefix_density_integrated_brier':cb,'P199_macro_prefix_integrated_brier':pb,'macro_Brier_reduction_vs_P199':br,'macro_prefix_density_calibration_error':ce,'P199_macro_prefix_calibration_error':pe,'macro_calibration_error_reduction_vs_P199':cr,'final_four_horizon_density_brier':float(np.mean((candidate[:,-1]-truth[:,-1])**2)),'P199_final_four_horizon_brier':float(np.mean((baseline[:,-1]-truth[:,-1])**2))},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
