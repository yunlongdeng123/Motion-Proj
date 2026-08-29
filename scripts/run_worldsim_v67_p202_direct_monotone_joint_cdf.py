"""Train a direct monotone seven-budget CDF for the four-horizon joint event."""

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


class DirectMonotoneJointCDF(nn.Module):
    def __init__(self,input_count,hidden_dimensions,budget_count):
        super().__init__();layers=[];width=input_count
        for hidden in hidden_dimensions:layers.extend((nn.Linear(width,int(hidden)),nn.SiLU()));width=int(hidden)
        layers.append(nn.Linear(width,budget_count));self.network=nn.Sequential(*layers)
    def forward(self,x):
        raw=self.network(x);logits=torch.cat((raw[:,:1],raw[:,:1]+torch.cumsum(functional.softplus(raw[:,1:]),dim=1)),dim=1);return torch.sigmoid(logits)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/"worldsim_v67"/c["task_id"]/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/"resolved.yaml").write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c["seed"]));torch.cuda.reset_peak_memory_stats()
    ens=torch.load(a.runs_root/c["frozen_p126"]["run"]/c["frozen_p126"]["artifact"],map_location="cuda");members=[]
    for state in ens["member_state_dicts"]:
        member=DirectionalActorGaussian(20,ens["hidden_dimensions"]).cuda();member.load_state_dict(state);members.append(member.eval())
    density,frozen_density=_load_density(a.runs_root/c["frozen_p182"]["run"]/c["frozen_p182"]["artifact"]);p199=torch.load(a.runs_root/c["frozen_p199"]["run"]/c["frozen_p199"]["artifact"],map_location="cuda");copula=JointHorizonCopula(8,p199["hidden_dimensions"],4).cuda();copula.load_state_dict(p199["model_state_dict"]);copula.eval()
    with np.load(a.runs_root/c["source_rows"]["run"]/c["source_rows"]["artifact"],allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
    horizons=np.asarray(c["horizons_seconds"],dtype=np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c["boundary_state_cost"]["clearance_floor_m"])),horizons);budgets=np.asarray(c["reliability_budgets"],dtype=np.float32);norms=tuple(frozen_density["norms"]);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),horizons[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],axis=1);independent=np.prod(marginal,axis=1);raw=np.concatenate((scores,clearances,np.log(np.clip(independent,1e-5,1-1e-5)/(1-np.clip(independent,1e-5,1-1e-5)))),axis=1).astype(np.float32);target=np.all(costs[:,:,None]<=budgets[None,None,:],axis=1).astype(np.float32);split=c["split"];dev=scenes%int(split["development_scene_modulus"])==int(split["development_scene_remainder"]);train=~dev;mean=raw[train].mean(0);scale=raw[train].std(0).clip(1e-6);x=torch.from_numpy(((raw-mean)/scale).astype(np.float32)).cuda();y=torch.from_numpy(target).cuda();indices=torch.from_numpy(np.flatnonzero(train)).cuda();m=c["model"];model=DirectMonotoneJointCDF(raw.shape[1],m["hidden_dimensions"],len(budgets)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=float(m["learning_rate"]),weight_decay=float(m["weight_decay"]));last=0.
    for step in range(int(m["training_steps"])):
        idx=indices[torch.randint(len(indices),(int(m["batch_size"]),),device="cuda")];prob=model(x[idx]);loss=torch.mean((prob-y[idx])**2);opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
        if step%500==0:print(f'P202 direct joint step={step+1} brier={last:.6f}',flush=True)
    dev_idx=torch.from_numpy(np.flatnonzero(dev)).cuda();model.eval()
    with torch.no_grad():candidate=model(x[dev_idx]).cpu().numpy()
    copula_features=torch.from_numpy(((np.concatenate((scores,clearances),axis=1)-np.asarray(p199["feature_mean"]))/np.asarray(p199["feature_scale"])).astype(np.float32)).cuda();p199_candidate=_joint_probabilities(copula,copula_features[dev_idx],torch.from_numpy(marginal[dev].astype(np.float32)).cuda(),int(c["evaluation"]["p199_monte_carlo_samples"]),int(c["seed"]));truth=target[dev];cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((p199_candidate-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(p199_candidate.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={"integrated_Brier_strictly_better_than_P199":cb<pb,"minimum_calibration_error_reduction_vs_P199":cr>=float(c["decision"]["minimum_calibration_error_reduction_vs_P199"])};verdict=c["verdict_on_pass"] if all(checks.values()) else c["verdict_on_failure"]
    artifact={"model_state_dict":model.state_dict(),"feature_mean":mean,"feature_scale":scale,"hidden_dimensions":m["hidden_dimensions"],"horizons_seconds":horizons,"reliability_budgets":budgets};torch.save(artifact,d/c["model_artifact"]);summary={"schema_version":c["output_schema_version"],"task_id":c["task_id"],"hypothesis_id":c["hypothesis_id"],"status":"done","verdict":verdict,"role":c["role"],"training":{"count":int(train.sum()),"final_batch_brier":last},"development":{"count":int(dev.sum()),"direct_integrated_brier":cb,"P199_integrated_brier":pb,"Brier_reduction_vs_P199":br,"direct_mean_absolute_reliability_error":ce,"P199_mean_absolute_reliability_error":pe,"calibration_error_reduction_vs_P199":cr},"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":c["claim_boundary"]};(d/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(d/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n");print(json.dumps({"run_dir":str(d),**summary},indent=2))


if __name__=="__main__":main()
