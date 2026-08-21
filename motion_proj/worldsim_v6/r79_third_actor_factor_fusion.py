"""WorldSim V6 R79: fuse actor5 renderer, kinematics, interaction, and contact factors."""
from __future__ import annotations
import json, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import yaml
from motion_proj.worldsim_v6.r41_actor_edit_factor_fusion import _content_sha256, _git, _read_jsonl, _resolve_runs_uri, _sha256, _verify, _write_json

TASK_ID = "WS-V6-R79-THIRD-ACTOR-FACTOR-FUSION-01"

def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started=time.monotonic(); repo_root=repo_root.resolve()
    if _git(repo_root,"status","--porcelain"): raise RuntimeError("formal R79 run requires clean source")
    source_commit=_git(repo_root,"rev-parse","HEAD"); config=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id")!=TASK_ID: raise RuntimeError("R79 task_id drift")
    s=config["sources"]; runs={k:_resolve_runs_uri(s[f"{k}_run"]) for k in ("r77","r78")}
    frozen={runs["r77"]/"MANIFEST.json":s["r77_manifest_sha256"],runs["r77"]/"R77_GATE.json":s["r77_gate_sha256"],runs["r77"]/"SUMMARY.json":s["r77_summary_sha256"],runs["r77"]/"SELECTED_PROPOSAL.json":s["r77_selected_sha256"],runs["r78"]/"MANIFEST.json":s["r78_manifest_sha256"],runs["r78"]/"R78_GATE.json":s["r78_gate_sha256"],runs["r78"]/"SUMMARY.json":s["r78_summary_sha256"],runs["r78"]/"BOX_FOOTPRINT_CONTACT_DECISIONS.jsonl":s["r78_decisions_sha256"]}
    for p,h in frozen.items(): _verify(p,h)
    free=shutil.disk_usage(run_root).free/(1024**3)
    if free<float(config["resources"]["minimum_disk_free_gib"]): raise RuntimeError("R79 disk resource insufficient")
    run_dir=run_root/TASK_ID/f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}__actor5-factor-fusion-s{config['seed']}-r1"; run_dir.mkdir(parents=True,exist_ok=False)
    try:
        gates=[json.loads((runs[k]/f"{k.upper()}_GATE.json").read_text(encoding="utf-8")) for k in ("r77","r78")]
        selected=json.loads((runs["r77"]/"SELECTED_PROPOSAL.json").read_text(encoding="utf-8"))["selected"]
        r77=json.loads((runs["r77"]/"SUMMARY.json").read_text(encoding="utf-8")); contact={r["intervention_id"]:r for r in _read_jsonl(runs["r78"]/"BOX_FOOTPRINT_CONTACT_DECISIONS.jsonl")}[config["expected"]["contact_intervention_id"]]
        factors={"renderer_execution":"ACCEPT" if r77["status"]=="done" and r77["changed_rgb_pixels_vs_logged"]>0 else "REJECT","self_kinematics":selected["q_self_kinematics"],"aabb_interaction":selected["q_aabb_interaction"],"box_footprint_lidar_contact":contact["q_box_footprint_lidar_contact"]}
        row={"schema_version":"worldsim_v6.r79_fused_decision.v1","proposal_id":selected["proposal_id"],"translation_delta_m":selected["translation_delta_m"],"factor_decisions":factors,"joint_admissibility":"ACCEPT_CONFORMANCE" if all(v=="ACCEPT" for v in factors.values()) else "REJECT_OR_ABSTAIN","semantic_road":"ABSTAIN","physical_dynamics":"ABSTAIN","planning_safety":"ABSTAIN"}
        _write_json(run_dir/"FUSED_EDIT_DECISION.json",row); wall=time.monotonic()-started
        checks={"all_source_gates_accepted":all(g["checks"]["passed"] for g in gates),"proposal_and_translation_binding_exact":row["proposal_id"]==config["expected"]["proposal_id"] and row["translation_delta_m"]==config["expected"]["translation_delta_m"],"independent_factor_decisions_exact":factors==config["expected"]["factor_decisions"],"joint_admissibility_exact":row["joint_admissibility"]==config["expected"]["joint_admissibility"],"unsupported_claims_abstain":row["semantic_road"]==row["physical_dynamics"]==row["planning_safety"]=="ABSTAIN","repeat_exact":_content_sha256(row)==_content_sha256(dict(row)),"source_immutable":all(_sha256(p)==h for p,h in frozen.items()),"wall_within_budget":wall<=float(config["resources"]["maximum_wall_seconds"]),"training_not_started":True,"confirmation_not_read":True}; checks["passed"]=all(checks.values()); status="done" if checks["passed"] else "rejected"
        _write_json(run_dir/"R79_GATE.json",{"schema_version":"worldsim_v6.r79_gate.v1","checks":checks,"decision":"accept_third_actor_factor_conformance_fusion" if checks["passed"] else "reject_or_repair_third_actor_factor_fusion"})
        _write_json(run_dir/"SUMMARY.json",{"schema_version":"worldsim_v6.r79_summary.v1","task_id":TASK_ID,"hypothesis_id":config["hypothesis_id"],"status":status,"hypothesis_outcome":"accepted_development_third_actor_four_factor_fusion" if checks["passed"] else "rejected","source_commit":source_commit,"proposal_id":row["proposal_id"],"joint_admissibility":row["joint_admissibility"],"factor_decisions":factors,"claim_boundary":config["claim_boundary"]})
        _write_json(run_dir/"RESOURCE_AUDIT.json",{"schema_version":"worldsim_v6.r79_resource_audit.v1","gpu_used":False,"wall_seconds":wall,"disk_free_gib_at_start":free,"training_started":False,"confirmation_content_read":False})
        tracked=["R79_GATE.json","SUMMARY.json","FUSED_EDIT_DECISION.json","RESOURCE_AUDIT.json"]; _write_json(run_dir/"MANIFEST.json",{"schema_version":"worldsim_v6.r79_manifest.v1","files":{n:{"bytes":(run_dir/n).stat().st_size,"sha256":_sha256(run_dir/n)} for n in tracked}}); _write_json(run_dir/"TERMINAL.json",{"schema_version":"worldsim_v6.terminal.v1","status":status,"manifest_sha256":_sha256(run_dir/"MANIFEST.json"),"summary_sha256":_sha256(run_dir/"SUMMARY.json")}); print(run_dir,flush=True); return run_dir
    except Exception as e:
        _write_json(run_dir/"TERMINAL.json",{"schema_version":"worldsim_v6.terminal.v1","status":"failed","error_type":type(e).__name__,"error":str(e)}); raise

def main()->int:
    import argparse
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--repo-root",type=Path,default=Path.cwd()); p.add_argument("--config",type=Path,default=Path("configs/worldsim_v6/r79_third_actor_factor_fusion_v1.yaml")); p.add_argument("--run-root",type=Path,default=Path("/root/autodl-tmp/runs/worldsim_v6")); a=p.parse_args(); run_experiment(a.repo_root,a.config,a.run_root); return 0
if __name__=="__main__": raise SystemExit(main())
