"""Importance-weight prefix survival training using unlabeled target features."""

from __future__ import annotations
import argparse,json,math,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p214_prefix_survival_max_cost_density import PrefixSurvivalDensity,_cdf,_prefix_copula_probabilities


def _payload(path,members,ens,floor,h):
 with np.load(path,allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 return _align(_trajectory_payload(arrays,members,ens,floor),h)


def _all_cdf(model,tokens,masks,budgets):
 flat=tokens[:,None].expand(-1,4,-1,-1).reshape(-1,4,3);flat_masks=masks[None].expand(len(tokens),-1,-1).reshape(-1,4)
 return torch.from_numpy(_cdf(model,flat,flat_masks,budgets).reshape(len(tokens),4,-1)).cuda()


def _beta(prob,raw_a,raw_b,bias):
 p=prob.clamp(1e-5,1-1e-5);return torch.sigmoid(functional.softplus(raw_a)*torch.log(p)-functional.softplus(raw_b)*torch.log1p(-p)+bias[None,:,None])


def _weighted_nll(logits,means,scales,target,weight):
 z=(target[:,None]-means)/scales;logd=-.5*z.square()-torch.log(scales)-.5*math.log(2*math.pi);nll=-torch.logsumexp(functional.log_softmax(logits,1)+logd,1)
 return (nll*weight).sum()/weight.sum()


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 h=np.asarray(c['horizons_seconds'],np.float32);floor=float(c['boundary_state_cost']['clearance_floor_m']);ss,sc,sy,scene=_payload(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],members,ens,floor,h);ts,tc,ty,_=_payload(a.runs_root/c['target_rows']['run']/c['target_rows']['artifact'],members,ens,floor,h);source_raw=np.stack((ss,sc,np.broadcast_to(h[None],ss.shape)),2);target_raw=np.stack((ts,tc,np.broadcast_to(h[None],ts.shape)),2);split=c['split'];rem=scene%int(split['scene_modulus']);cal=rem==int(split['calibration_remainder']);train=(rem!=int(split['excluded_development_remainder']))&~cal;mean=source_raw[train].reshape(-1,3).mean(0);scale=source_raw[train].reshape(-1,3).std(0).clip(1e-6);source_tokens=torch.from_numpy(((source_raw-mean)/scale).astype(np.float32)).cuda();target_tokens=torch.from_numpy(((target_raw-mean)/scale).astype(np.float32)).cuda();source_domain=source_tokens[:,:,:2].reshape(len(source_tokens),-1);target_domain=target_tokens[:,:,:2].reshape(len(target_tokens),-1);dc=c['domain_classifier'];domain=nn.Sequential(nn.Linear(8,int(dc['hidden_dimension'])),nn.SiLU(),nn.Linear(int(dc['hidden_dimension']),1)).cuda();dop=torch.optim.Adam(domain.parameters(),lr=float(dc['learning_rate']));half=int(dc['batch_size'])//2
 for step in range(int(dc['steps'])):
  sx=source_domain[torch.randint(len(source_domain),(half,),device='cuda')];tx=target_domain[torch.randint(len(target_domain),(half,),device='cuda')];logits=domain(torch.cat((sx,tx))).squeeze(1);truth=torch.cat((torch.zeros(half,device='cuda'),torch.ones(half,device='cuda')));loss=functional.binary_cross_entropy_with_logits(logits,truth);dop.zero_grad(set_to_none=True);loss.backward();dop.step()
  if step%500==0:print(f'P217 domain step={step+1} bce={float(loss):.6f}',flush=True)
 with torch.no_grad():weight=torch.exp(domain(source_domain).squeeze(1)).clamp(float(dc['minimum_weight']),float(dc['maximum_weight']));weight/=weight.mean();domain_accuracy=float(((torch.sigmoid(domain(torch.cat((source_domain,target_domain))).squeeze(1))>=.5)==torch.cat((torch.zeros(len(source_domain),device='cuda'),torch.ones(len(target_domain),device='cuda'))).bool()).float().mean());ess=float(weight.sum().square()/weight.square().sum())
 prefix=np.maximum.accumulate(sy,1);target=torch.from_numpy(np.log1p(prefix).astype(np.float32)).cuda();masks=torch.tril(torch.ones((4,4),dtype=torch.bool,device='cuda'));train_idx=torch.from_numpy(np.flatnonzero(train)).cuda();m=c['density_model'];model=PrefixSurvivalDensity(int(m['component_count']),int(m['token_dimension']),m['head_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  trajectory=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];pi=torch.randint(4,(len(trajectory),),device='cuda');logits,means,scales=model(source_tokens[trajectory],masks[pi]);loss=_weighted_nll(logits,means,scales,target[trajectory,pi],weight[trajectory]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P217 density step={step+1} nll={last:.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);cal_raw=_all_cdf(model.eval(),source_tokens[torch.from_numpy(np.flatnonzero(cal)).cuda()],masks,budgets);cal_truth=torch.from_numpy((prefix[cal,:,None]<=budgets[None,None,:]).astype(np.float32)).cuda();cal_w=weight[torch.from_numpy(np.flatnonzero(cal)).cuda()];raw_a=nn.Parameter(torch.tensor(0.,device='cuda'));raw_b=nn.Parameter(torch.tensor(0.,device='cuda'));bias=nn.Parameter(torch.zeros(4,device='cuda'));co=torch.optim.Adam((raw_a,raw_b,bias),lr=float(c['calibrator']['learning_rate']));clast=0.
 for step in range(int(c['calibrator']['steps'])):
  out=_beta(cal_raw,raw_a,raw_b,bias);each=functional.binary_cross_entropy(out,cal_truth,reduction='none');loss=(each*cal_w[:,None,None]).sum()/(cal_w.sum()*28);co.zero_grad(set_to_none=True);loss.backward();co.step();clast=float(loss.detach())
  if step%300==0:print(f'P217 beta step={step+1} logloss={clast:.6f}',flush=True)
 candidate=_beta(_all_cdf(model.eval(),target_tokens,masks,budgets),raw_a,raw_b,bias).detach().cpu().numpy();marginal_density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(marginal_density,ts[:,i],np.full(len(ts),h[i],np.float32),tc[:,i],budgets,norms) for i in range(4)],1);raw=np.concatenate((ts,tc),1);features=torch.from_numpy(((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();baseline=_prefix_copula_probabilities(copula,features,torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c['evaluation']['monte_carlo_samples']),int(c['seed']));truth=np.maximum.accumulate(ty,1)[:,:,None]<=budgets[None,None,:];cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((baseline-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(baseline.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'target_macro_prefix_Brier_strictly_better_than_P199':cb<pb,'target_macro_calibration_error_noninferior_to_P199':cr>=float(c['decision']['minimum_macro_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'token_mean':mean,'token_scale':scale,'component_count':m['component_count'],'token_dimension':m['token_dimension'],'head_dimensions':m['head_dimensions'],'beta_raw_a':raw_a.detach().cpu(),'beta_raw_b':raw_b.detach().cpu(),'beta_prefix_bias':bias.detach().cpu()},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'source_density_trajectories':int(train.sum()),'source_calibration_trajectories':int(cal.sum()),'unlabeled_target_trajectories':int(len(ts)),'domain_accuracy':domain_accuracy,'importance_weight_min':float(weight.min()),'importance_weight_max':float(weight.max()),'importance_weight_effective_sample_size':ess,'final_density_nll':last,'final_calibration_logloss':clast},'target_development':{'macro_prefix_integrated_brier':cb,'P199_macro_prefix_integrated_brier':pb,'macro_Brier_reduction_vs_P199':br,'macro_prefix_calibration_error':ce,'P199_macro_prefix_calibration_error':pe,'macro_calibration_error_reduction_vs_P199':cr},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
