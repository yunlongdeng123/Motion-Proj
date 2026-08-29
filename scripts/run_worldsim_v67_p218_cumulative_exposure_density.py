"""Density of time-weighted cumulative visited-state cost."""

from __future__ import annotations
import argparse,json,math,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _mixture_nll
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p212_deepset_joint_max_cost_density import DeepSetMaxCostDensity,_cdf


@torch.no_grad()
def _factorized_exposure_cdf(marginal,copula,features,scores,clearances,h,budgets,intervals,norms,samples,steps,batch_size,seed):
 generator=torch.Generator(device='cuda').manual_seed(seed);outputs=[];budget=torch.from_numpy(budgets).cuda();dt=torch.from_numpy(intervals).cuda()
 for start in range(0,len(features),batch_size):
  x=features[start:start+batch_size];b=len(x);noise=torch.randn((b,samples,4),generator=generator,device='cuda');latent=torch.einsum('bij,bsj->bsi',copula.correlation_cholesky(x),noise);u=(.5*(1+torch.erf(latent/math.sqrt(2)))).clamp(1e-6,1-1e-6);condition=np.stack(((scores[start:start+b]-norms[0])/norms[1],(np.broadcast_to(h[None],(b,4))-norms[2])/norms[3],(clearances[start:start+b]-norms[4])/norms[5]),2).astype(np.float32);logits,means,scales=marginal(torch.from_numpy(condition).cuda().reshape(-1,3));k=logits.shape[1];weights=functional.softmax(logits,1).reshape(b,4,k);means=means.reshape(b,4,k);scales=scales.reshape(b,4,k);lo=(means-8*scales).min(2).values[:,None].expand(-1,samples,-1).clone();hi=(means+8*scales).max(2).values[:,None].expand(-1,samples,-1).clone()
  for _ in range(steps):
   mid=(lo+hi)/2;z=(mid[:,:,:,None]-means[:,None])/scales[:,None];cdf=torch.sum(weights[:,None]*(.5*(1+torch.erf(z/math.sqrt(2)))),3);lo=torch.where(cdf<u,mid,lo);hi=torch.where(cdf<u,hi,mid)
  cost=torch.expm1((lo+hi)/2).clamp_min(0);exposure=(cost*dt[None,None]).sum(2);outputs.append((exposure[:,:,None]<=budget[None,None]).float().mean(1).cpu().numpy())
 return np.concatenate(outputs)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True;ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 marginal,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);intervals=np.asarray(c['exposure_interval_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;raw_tokens=np.stack((scores,clearances,np.broadcast_to(h[None],scores.shape)),2);mean=raw_tokens[train].reshape(-1,3).mean(0);scale=raw_tokens[train].reshape(-1,3).std(0).clip(1e-6);tokens=torch.from_numpy(((raw_tokens-mean)/scale).astype(np.float32)).cuda();exposure=np.sum(costs*intervals[None],1);target=torch.from_numpy(np.log1p(exposure).astype(np.float32)).cuda();train_idx=torch.from_numpy(np.flatnonzero(train)).cuda();m=c['model'];model=DeepSetMaxCostDensity(int(m['component_count']),int(m['token_dimension']),m['head_dimensions']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));last=0.
 for step in range(int(m['steps'])):
  idx=train_idx[torch.randint(len(train_idx),(int(m['batch_size']),),device='cuda')];logits,means,scales=model(tokens[idx]);loss=_mixture_nll(logits,means,scales,target[idx]);opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P218 cumulative exposure density step={step+1} nll={last:.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);candidate=_cdf(model.eval(),tokens[torch.from_numpy(np.flatnonzero(dev)).cuda()],budgets);raw=np.concatenate((scores,clearances),1);features=torch.from_numpy(((raw[dev]-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();ev=c['evaluation'];baseline=_factorized_exposure_cdf(marginal,copula,features,scores[dev],clearances[dev],h,budgets,intervals,tuple(fd['norms']),int(ev['monte_carlo_samples']),int(ev['inverse_cdf_bisection_steps']),int(ev['batch_size']),int(c['seed']));truth=exposure[dev,None]<=budgets[None];cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((baseline-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(baseline.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'integrated_Brier_strictly_better_than_P199_factorized_exposure':cb<pb,'calibration_error_noninferior_to_P199_factorized_exposure':cr>=float(c['decision']['minimum_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'token_mean':mean,'token_scale':scale,'component_count':m['component_count'],'token_dimension':m['token_dimension'],'head_dimensions':m['head_dimensions'],'horizons_seconds':h,'exposure_interval_seconds':intervals},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'trajectory_count':int(train.sum()),'final_log_cumulative_exposure_nll':last},'development':{'trajectory_count':int(dev.sum()),'density_integrated_brier':cb,'P199_factorized_integrated_brier':pb,'Brier_reduction_vs_P199_factorized':br,'density_calibration_error':ce,'P199_factorized_calibration_error':pe,'calibration_error_reduction_vs_P199_factorized':cr},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
