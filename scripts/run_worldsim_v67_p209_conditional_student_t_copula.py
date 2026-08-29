"""Fit a fixed-df conditional Student-t copula over frozen P182 marginals."""

from __future__ import annotations
import argparse,json,math,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import t as scipy_t
import torch
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload,_variable_cdf


@torch.no_grad()
def _t_joint_probabilities(model,features,marginal,samples,seed,df):
 model.eval();threshold=torch.from_numpy(scipy_t.ppf(marginal.detach().cpu().numpy().clip(1e-4,1-1e-4),df=df).astype(np.float32)).cuda();outputs=[];generator=torch.Generator(device='cuda').manual_seed(seed)
 for start in range(0,len(features),512):
  x=features[start:start+512];bound=threshold[start:start+512];chol=model.correlation_cholesky(x);normal=torch.randn((len(x),samples,4),generator=generator,device='cuda');radial=torch.randn((len(x),samples,int(df)),generator=generator,device='cuda').square().sum(2);draws=torch.einsum('bij,bsj->bsi',chol,normal)/torch.sqrt(radial[:,:,None]/float(df));outputs.append((draws[:,:,:,None]<=bound[:,None,:,:]).all(2).float().mean(1).cpu().numpy())
 return np.concatenate(outputs)


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats()
 ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');control=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();control.load_state_dict(p199['model_state_dict']);control.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);raw=np.concatenate((scores,clearances),1);features=torch.from_numpy(((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();norms=tuple(fd['norms']);clip=float(c['evaluation']['probability_clip']);u=np.stack([_variable_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],costs[:,i],norms) for i in range(4)],1);m=c['model'];nu=float(m['degrees_of_freedom']);q=torch.from_numpy(scipy_t.ppf(np.clip(u,clip,1-clip),df=nu).astype(np.float32)).cuda();split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;indices=torch.from_numpy(np.flatnonzero(train)).cuda();model=JointHorizonCopula(8,m['hidden_dimensions'],4).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m['learning_rate']),weight_decay=float(m['weight_decay']));batch=int(m['batch_size']);dimension=4;multi_constant=math.lgamma((nu+dimension)/2)-math.lgamma(nu/2)-dimension/2*math.log(nu*math.pi);uni_constant=dimension*(math.lgamma((nu+1)/2)-math.lgamma(nu/2)-.5*math.log(nu*math.pi));last=0.
 for step in range(int(m['steps'])):
  idx=indices[torch.randint(len(indices),(batch,),device='cuda')];chol=model.correlation_cholesky(features[idx]);target=q[idx];solved=torch.cholesky_solve(target[:,:,None],chol).squeeze(2);quad=(target*solved).sum(1);logdet=2*torch.log(torch.diagonal(chol,dim1=1,dim2=2)).sum(1);multi=multi_constant-.5*logdet-(nu+dimension)/2*torch.log1p(quad/nu);uni=uni_constant-(nu+1)/2*torch.log1p(target.square()/nu).sum(1);loss=-(multi-uni).mean();opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P209 conditional t copula step={step+1} copula_nll={last:.6f}',flush=True)
 budgets=np.asarray(c['reliability_budgets'],np.float32);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);dev_idx=torch.from_numpy(np.flatnonzero(dev)).cuda();marginal_tensor=torch.from_numpy(marginal[dev].astype(np.float32)).cuda();candidate=_t_joint_probabilities(model,features[dev_idx],marginal_tensor,int(c['evaluation']['monte_carlo_samples']),int(c['seed']),nu);baseline=_joint_probabilities(control,features[dev_idx],marginal_tensor,int(c['evaluation']['monte_carlo_samples']),int(c['seed']));truth=np.all(costs[dev,:,None]<=budgets[None,None,:],1);cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((baseline-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(baseline.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'integrated_Brier_strictly_better_than_P199':cb<pb,'calibration_error_noninferior_to_P199':cr>=float(c['decision']['minimum_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'model_state_dict':model.state_dict(),'degrees_of_freedom':nu,'hidden_dimensions':m['hidden_dimensions'],'feature_mean':p199['feature_mean'],'feature_scale':p199['feature_scale'],'base_marginal':c['frozen_p182']},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'final_student_t_copula_nll':last,'degrees_of_freedom':nu},'development':{'student_t_integrated_brier':cb,'P199_integrated_brier':pb,'Brier_reduction_vs_P199':br,'student_t_mean_absolute_reliability_error':ce,'P199_mean_absolute_reliability_error':pe,'calibration_error_reduction_vs_P199':cr},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
