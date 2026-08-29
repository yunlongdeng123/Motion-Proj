"""Fit a shared rank-preserving beta calibration map over frozen P199 probabilities."""

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


class MonotoneBetaCalibration(nn.Module):
    def __init__(self):
        super().__init__();identity_raw=float(np.log(np.expm1(1.0)));self.raw_a=nn.Parameter(torch.tensor(identity_raw));self.raw_b=nn.Parameter(torch.tensor(identity_raw));self.intercept=nn.Parameter(torch.tensor(0.0))
    def forward(self,p):
        p=p.clamp(1e-5,1-1e-5);a=functional.softplus(self.raw_a);b=functional.softplus(self.raw_b);return torch.sigmoid(a*torch.log(p)-b*torch.log1p(-p)+self.intercept)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/"worldsim_v67"/c["task_id"]/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/"resolved.yaml").write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c["seed"]));torch.cuda.reset_peak_memory_stats()
    ens=torch.load(a.runs_root/c["frozen_p126"]["run"]/c["frozen_p126"]["artifact"],map_location="cuda");members=[]
    for state in ens["member_state_dicts"]:
        member=DirectionalActorGaussian(20,ens["hidden_dimensions"]).cuda();member.load_state_dict(state);members.append(member.eval())
    density,frozen_density=_load_density(a.runs_root/c["frozen_p182"]["run"]/c["frozen_p182"]["artifact"]);p199=torch.load(a.runs_root/c["frozen_p199"]["run"]/c["frozen_p199"]["artifact"],map_location="cuda");copula=JointHorizonCopula(8,p199["hidden_dimensions"],4).cuda();copula.load_state_dict(p199["model_state_dict"]);copula.eval()
    with np.load(a.runs_root/c["source_rows"]["run"]/c["source_rows"]["artifact"],allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
    horizons=np.asarray(c["horizons_seconds"],dtype=np.float32);scores,clearances,costs,scenes=_align(_trajectory_payload(arrays,members,ens,float(c["boundary_state_cost"]["clearance_floor_m"])),horizons);budgets=np.asarray(c["reliability_budgets"],dtype=np.float32);norms=tuple(frozen_density["norms"]);marginal=np.stack([_predict_cdf(density,scores[:,i],np.full(len(scores),horizons[i],np.float32),clearances[:,i],budgets,norms) for i in range(4)],axis=1);features=torch.from_numpy(((np.concatenate((scores,clearances),axis=1)-np.asarray(p199["feature_mean"]))/np.asarray(p199["feature_scale"])).astype(np.float32)).cuda();base=_joint_probabilities(copula,features,torch.from_numpy(marginal.astype(np.float32)).cuda(),int(c["calibrator"]["monte_carlo_samples"]),int(c["seed"]));target=np.all(costs[:,:,None]<=budgets[None,None,:],axis=1).astype(np.float32);split=c["split"];dev=scenes%int(split["development_scene_modulus"])==int(split["development_scene_remainder"]);train=~dev;base_tensor=torch.from_numpy(base).cuda();target_tensor=torch.from_numpy(target).cuda();flat_index=torch.from_numpy(np.flatnonzero(np.repeat(train,len(budgets)))).cuda();model=MonotoneBetaCalibration().cuda();opt=torch.optim.Adam(model.parameters(),lr=float(c["calibrator"]["learning_rate"]));flat_p=base_tensor.reshape(-1);flat_y=target_tensor.reshape(-1);last=0.
    for step in range(int(c["calibrator"]["training_steps"])):
        idx=flat_index[torch.randint(len(flat_index),(int(c["calibrator"]["batch_size"]),),device="cuda")];pred=model(flat_p[idx]);loss=torch.mean((pred-flat_y[idx])**2);opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
        if step%500==0:print(f'P203 beta step={step+1} brier={last:.6f}',flush=True)
    with torch.no_grad():candidate=model(base_tensor[torch.from_numpy(np.flatnonzero(dev)).cuda()]).cpu().numpy();aa=float(functional.softplus(model.raw_a));bbeta=float(functional.softplus(model.raw_b));intercept=float(model.intercept)
    truth=target[dev];control=base[dev];cb=float(np.mean((candidate-truth)**2));pb=float(np.mean((control-truth)**2));ce=float(np.mean(np.abs(candidate.mean(0)-truth.mean(0))));pe=float(np.mean(np.abs(control.mean(0)-truth.mean(0))));br=(pb-cb)/pb;cr=(pe-ce)/max(pe,1e-12);checks={"integrated_Brier_strictly_better_than_P199":cb<pb,"minimum_calibration_error_reduction_vs_P199":cr>=float(c["decision"]["minimum_calibration_error_reduction_vs_P199"])};verdict=c["verdict_on_pass"] if all(checks.values()) else c["verdict_on_failure"]
    artifact={"model_state_dict":model.state_dict(),"a":aa,"b":bbeta,"intercept":intercept,"base_model":c["frozen_p199"]};torch.save(artifact,d/c["model_artifact"]);summary={"schema_version":c["output_schema_version"],"task_id":c["task_id"],"hypothesis_id":c["hypothesis_id"],"status":"done","verdict":verdict,"role":c["role"],"calibrator":{"a":aa,"b":bbeta,"intercept":intercept,"final_batch_brier":last},"development":{"count":int(dev.sum()),"calibrated_integrated_brier":cb,"P199_integrated_brier":pb,"Brier_reduction_vs_P199":br,"calibrated_mean_absolute_reliability_error":ce,"P199_mean_absolute_reliability_error":pe,"calibration_error_reduction_vs_P199":cr},"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":c["claim_boundary"]};(d/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(d/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n");print(json.dumps({"run_dir":str(d),**summary},indent=2))


if __name__=="__main__":main()
