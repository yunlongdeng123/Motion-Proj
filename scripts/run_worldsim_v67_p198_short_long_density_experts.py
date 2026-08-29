"""Fine-tune parameter-isolated short/long density experts on source horizons."""

from __future__ import annotations
import argparse,json,resource,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import yaml

from scripts.run_worldsim_v67_p109_directional_actor_uncertainty import DirectionalActorGaussian
from scripts.run_worldsim_v67_p120_continuous_boundary_state_cost import _continuous_cost
from scripts.run_worldsim_v67_p127_ensemble_continuous_selection import _ensemble_trajectory_score
from scripts.run_worldsim_v67_p166_monotone_expected_cost_calibration import _trajectory_horizon
from scripts.run_worldsim_v67_p178_clearance_conditioned_reliability_cdf import _trajectory_clearance
from scripts.run_worldsim_v67_p182_log_cost_mixture_density import LogCostMixtureDensity,_mixture_nll,_predict_cdf


def load_density(path):
    f=torch.load(path,map_location="cuda");m=LogCostMixtureDensity(int(f["component_count"]),list(f["hidden_dimensions"])).cuda();m.load_state_dict(f["model_state_dict"]);return m,f


def prepare(arrays,members,ens,floor):
    score,scenes=_ensemble_trajectory_score(arrays,members,np.asarray(ens["feature_mean"],dtype=np.float32),np.asarray(ens["feature_scale"],dtype=np.float32),np.asarray(ens["target_mean"],dtype=np.float32),np.asarray(ens["target_scale"],dtype=np.float32));cost,cost_scenes=_continuous_cost(arrays,floor)
    if not np.array_equal(scenes,cost_scenes):raise RuntimeError("P198 grouping mismatch")
    return score,_trajectory_horizon(arrays),_trajectory_clearance(arrays),cost,scenes


def condition(score,horizon,clearance,norms):
    return np.stack(((score-norms[0])/norms[1],(horizon-norms[2])/norms[3],(clearance-norms[4])/norms[5]),axis=1).astype(np.float32)


def train(model,x,y,index_sampler,steps,batch,lr,wd,label):
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd);last=0.
    for step in range(steps):
        idx=index_sampler(batch);logits,means,scales=model(x[idx]);loss=_mixture_nll(logits,means,scales,y[idx]);opt.zero_grad();loss.backward();opt.step();last=float(loss.detach())
        if step%500==0:print(f"P198 {label} step={step+1} nll={last:.6f}",flush=True)
    return last


def routed(short,long,score,horizon,clearance,budgets,norms,transition):
    a=_predict_cdf(short,score,horizon,clearance,budgets,norms);b=_predict_cdf(long,score,horizon,clearance,budgets,norms);lo=float(transition["short_end_seconds"]);hi=float(transition["long_start_seconds"]);w=np.clip((horizon-lo)/(hi-lo),0,1).astype(np.float32);return (1-w[:,None])*a+w[:,None]*b,w


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--runs-root",type=Path,required=True);p.add_argument("--run-id",required=True);a=p.parse_args();c=yaml.safe_load(a.config.read_text());d=a.runs_root/"worldsim_v67"/c["task_id"]/a.run_id;d.mkdir(parents=True,exist_ok=False);(d/"resolved.yaml").write_text(yaml.safe_dump(c,sort_keys=False));started=time.monotonic();torch.manual_seed(int(c["seed"]));torch.cuda.reset_peak_memory_stats()
    ens=torch.load(a.runs_root/c["frozen_p126"]["run"]/c["frozen_p126"]["artifact"],map_location="cuda");members=[]
    for state in ens["member_state_dicts"]:
        m=DirectionalActorGaussian(20,ens["hidden_dimensions"]).cuda();m.load_state_dict(state);members.append(m.eval())
    short,f182=load_density(a.runs_root/c["frozen_p182"]["run"]/c["frozen_p182"]["artifact"]);long,f192=load_density(a.runs_root/c["frozen_p192"]["run"]/c["frozen_p192"]["artifact"]);norms=tuple(f182["norms"])
    with np.load(a.runs_root/c["source_rows"]["run"]/c["source_rows"]["artifact"],allow_pickle=False) as z:source={n:z[n] for n in z.files}
    score,horizon,clearance,cost,scenes=prepare(source,members,ens,float(c["boundary_state_cost"]["clearance_floor_m"]));x=torch.from_numpy(condition(score,horizon,clearance,norms)).cuda();y=torch.from_numpy(np.log1p(cost).astype(np.float32)).cuda();short_idx=np.flatnonzero(horizon<=1.5+1e-4);long_idx=np.flatnonzero(horizon>=2.5-1e-4);short_tensor=torch.from_numpy(short_idx).cuda();groups=[long_idx[scenes[long_idx]==s] for s in np.unique(scenes[long_idx])];mx=max(map(len,groups));table=np.zeros((len(groups),mx),np.int64);length=np.asarray(list(map(len,groups)),np.int64)
    for i,g in enumerate(groups):table[i,:len(g)]=g
    table=torch.from_numpy(table).cuda();length=torch.from_numpy(length).cuda()
    short_sampler=lambda n:short_tensor[torch.randint(len(short_tensor),(n,),device="cuda")]
    def long_sampler(n):
        scene=torch.randint(len(length),(n,),device="cuda");local=torch.floor(torch.rand((n,),device="cuda")*length[scene].float()).long();return table[scene,local]
    t=c["training"];short_nll=train(short,x,y,short_sampler,int(t["steps_per_expert"]),int(t["batch_size"]),float(t["learning_rate"]),float(t["weight_decay"]),"short");long_nll=train(long,x,y,long_sampler,int(t["steps_per_expert"]),int(t["batch_size"]),float(t["learning_rate"]),float(t["weight_decay"]),"long");short.eval();long.eval();budgets=np.asarray(c["reliability_budgets"],dtype=np.float32);evaluations={}
    p182,_=load_density(a.runs_root/c["frozen_p182"]["run"]/c["frozen_p182"]["artifact"]);p182.eval()
    for cohort in c["decision_cohorts"]:
        with np.load(a.runs_root/cohort["run"]/cohort["artifact"],allow_pickle=False) as z:arrays={n:z[n] for n in z.files}
        s,h,cl,cost,_=prepare(arrays,members,ens,float(c["boundary_state_cost"]["clearance_floor_m"]));pred,w=routed(short,long,s,h,cl,budgets,norms,c["experts"]["transition"]);base=_predict_cdf(p182,s,h,cl,budgets,norms);target=cost[:,None]<=budgets[None];pb=float(np.mean((pred-target)**2));bb=float(np.mean((base-target)**2));pe=float(np.mean(np.abs(pred.mean(0)-target.mean(0))));be=float(np.mean(np.abs(base.mean(0)-target.mean(0))));evaluations[cohort["name"]]={"trajectory_count":int(len(cost)),"expert_integrated_brier":pb,"p182_integrated_brier":bb,"brier_change_vs_p182":(pb-bb)/bb,"expert_mean_absolute_reliability_error":pe,"p182_mean_absolute_reliability_error":be,"calibration_error_reduction_vs_p182":(be-pe)/max(be,1e-12),"mean_long_expert_weight":float(w.mean())}
    cr=[r["calibration_error_reduction_vs_p182"] for r in evaluations.values()];checks={"brier_noninferior_to_p182_every_cohort":all(r["expert_integrated_brier"]<=r["p182_integrated_brier"] for r in evaluations.values()),"minimum_mean_calibration_error_reduction_vs_p182":float(np.mean(cr))>=float(c["decision"]["minimum_mean_calibration_error_reduction_vs_p182"])};verdict=c["verdict_on_pass"] if all(checks.values()) else c["verdict_on_failure"]
    artifact={"short_state_dict":short.state_dict(),"long_state_dict":long.state_dict(),"component_count":f182["component_count"],"hidden_dimensions":f182["hidden_dimensions"],"norms":norms,"transition":c["experts"]["transition"]};torch.save(artifact,d/c["model_artifact"]);summary={"schema_version":c["output_schema_version"],"task_id":c["task_id"],"hypothesis_id":c["hypothesis_id"],"status":"done","verdict":verdict,"role":c["role"],"training":{"short_trajectory_count":int(len(short_idx)),"long_trajectory_count":int(len(long_idx)),"short_final_nll":short_nll,"long_final_nll":long_nll},"consumed_development_evaluations":evaluations,"mean_calibration_error_reduction_vs_p182":float(np.mean(cr)),"decision_checks":checks,"resources":{"gpu":torch.cuda.get_device_name(0),"peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"wall_seconds":time.monotonic()-started},"claim_boundary":c["claim_boundary"]};(d/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");(d/"status.json").write_text(json.dumps({"status":"done","completed_at_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n");print(json.dumps({"run_dir":str(d),**summary},indent=2))


if __name__=="__main__":main()
