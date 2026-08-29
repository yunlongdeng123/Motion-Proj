"""Pool frozen P199 and P210 joint probabilities with one proper-score weight."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import _predict_cdf
from scripts.run_worldsim_v67_p199_joint_horizon_reliability_copula import JointHorizonCopula,_align,_joint_probabilities,_load_density,_trajectory_payload
from scripts.run_worldsim_v67_p210_joint_max_cost_density import JointMaxCostDensity,_max_cdf


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--runs-root',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/'worldsim_v67'/c['task_id']/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/'resolved.yaml').write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c['seed']));torch.cuda.reset_peak_memory_stats()
 ens=torch.load(a.runs_root/c['frozen_p126']['run']/c['frozen_p126']['artifact'],map_location='cuda');members=[]
 for state in ens['member_state_dicts']:
  member=DirectionalActorGaussian(20,ens['hidden_dimensions']).cuda();member.load_state_dict(state);members.append(member.eval())
 marginal_density,fd=_load_density(a.runs_root/c['frozen_p182']['run']/c['frozen_p182']['artifact']);p199=torch.load(a.runs_root/c['frozen_p199']['run']/c['frozen_p199']['artifact'],map_location='cuda');copula=JointHorizonCopula(8,p199['hidden_dimensions'],4).cuda();copula.load_state_dict(p199['model_state_dict']);copula.eval();p210=torch.load(a.runs_root/c['frozen_p210']['run']/c['frozen_p210']['artifact'],map_location='cuda');max_density=JointMaxCostDensity(8,int(p210['component_count']),p210['hidden_dimensions']).cuda();max_density.load_state_dict(p210['model_state_dict']);max_density.eval()
 with np.load(a.runs_root/c['source_rows']['run']/c['source_rows']['artifact'],allow_pickle=False) as loaded:arrays={n:loaded[n] for n in loaded.files}
 h=np.asarray(c['horizons_seconds'],np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c['boundary_state_cost']['clearance_floor_m'])),h);raw=np.concatenate((scores,clearances),1);split=c['split'];dev=scenes%int(split['development_scene_modulus'])==int(split['development_scene_remainder']);train=~dev;budgets=np.asarray(c['reliability_budgets'],np.float32);p210_features=torch.from_numpy(((raw-np.asarray(p210['feature_mean']))/np.asarray(p210['feature_scale'])).astype(np.float32)).cuda();max_probability=_max_cdf(max_density,p210_features,budgets);p199_features=torch.from_numpy(((raw-np.asarray(p199['feature_mean']))/np.asarray(p199['feature_scale'])).astype(np.float32)).cuda();norms=tuple(fd['norms']);marginal=np.stack([_predict_cdf(marginal_density,scores[:,i],np.full(len(scores),h[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],1);copula_probability=_joint_probabilities(copula,p199_features,torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c['evaluation']['monte_carlo_samples']),int(c['seed']));truth=np.all(costs[:,:,None]<=budgets[None,None,:],1).astype(np.float32);p199_tensor=torch.from_numpy(copula_probability[train]).cuda();p210_tensor=torch.from_numpy(max_probability[train]).cuda();truth_tensor=torch.from_numpy(truth[train]).cuda();logit=nn.Parameter(torch.tensor(float(c['training']['initial_P199_logit']),device='cuda'));opt=torch.optim.Adam([logit],lr=float(c['training']['learning_rate']));batch=int(c['training']['batch_size']);last=0.
 for step in range(int(c['training']['steps'])):
  idx=torch.randint(len(p199_tensor),(batch,),device='cuda');weight=torch.sigmoid(logit);candidate=weight*p199_tensor[idx]+(1-weight)*p210_tensor[idx];loss=(candidate-truth_tensor[idx]).square().mean();opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
  if step%500==0:print(f'P211 linear pool step={step+1} brier={last:.6f} P199_weight={float(weight):.6f}',flush=True)
 weight=float(torch.sigmoid(logit).detach());candidate=weight*copula_probability[dev]+(1-weight)*max_probability[dev];baseline=copula_probability[dev];target=truth[dev];cb=float(np.mean((candidate-target)**2));pb=float(np.mean((baseline-target)**2));ce=float(np.mean(np.abs(candidate.mean(0)-target.mean(0))));pe=float(np.mean(np.abs(baseline.mean(0)-target.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={'integrated_Brier_strictly_better_than_P199':cb<pb,'calibration_error_noninferior_to_P199':cr>=float(c['decision']['minimum_calibration_error_reduction_vs_P199'])};verdict=c['verdict_on_pass'] if all(checks.values()) else c['verdict_on_failure'];torch.save({'P199_weight':weight,'components':c['frozen_p199']|{'P210':c['frozen_p210']}},d/c['model_artifact']);summary={'schema_version':c['output_schema_version'],'task_id':c['task_id'],'hypothesis_id':c['hypothesis_id'],'status':'done','verdict':verdict,'role':c['role'],'training':{'final_batch_Brier':last,'P199_weight':weight,'P210_weight':1-weight},'development':{'pooled_integrated_brier':cb,'P199_integrated_brier':pb,'Brier_reduction_vs_P199':br,'pooled_mean_absolute_reliability_error':ce,'P199_mean_absolute_reliability_error':pe,'calibration_error_reduction_vs_P199':cr},'decision_checks':checks,'resources':{'gpu':torch.cuda.get_device_name(0),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,'wall_seconds':time.monotonic()-started},'claim_boundary':c['claim_boundary']};(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'status.json').write_text(json.dumps({'status':'done','completed_at_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');print(json.dumps({'run_dir':str(d),**summary},indent=2))


if __name__=='__main__':main()
