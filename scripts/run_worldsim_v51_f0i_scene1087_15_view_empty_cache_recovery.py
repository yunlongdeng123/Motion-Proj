#!/usr/bin/env python3
"""执行 F0i scene-1087 15-view empty-cache recovery。"""

from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from motion_proj.worldsim_v51.protocol import ProtocolError
from scripts.run_worldsim_v51_f0a_environment_one_view_smoke import _git_at
from scripts.run_worldsim_v51_f0b_three_view_association_parity import _load_yaml, _verify, repository_source_identity
from scripts.run_worldsim_v51_f0c_upstream_batch_association_repeatability import _nvidia_total_mib
from scripts.run_worldsim_v51_f0g_target_tensor_allocator_instrumentation import _run_trace_attempt
from scripts.run_worldsim_v51_h_uplift import ResourceMonitor, _inventory, _nvidia_used_mib, _utc_now, _write_json, _write_jsonl, _write_text

SCHEMA="worldsim_v51_stage_f_f0i_scene1087_15_view_empty_cache_recovery_v1"
TASK_ID="WS-V51-M1-F-IDENTITY-EMBEDDING-01"

def _validate_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    c=_load_yaml(path)
    if c.get("schema_version")!=SCHEMA or c.get("task_id")!=TASK_ID: raise ProtocolError("F0i config drift")
    a=c["authorization"]["f0h_freeze"]; f=_load_yaml(_verify(PROJECT/a["path"],a["sha256"],"F0h freeze",int(a["bytes"])))
    if f.get("status")!=a["required_status"] or f["interpretation"].get("failure")!=a["required_failure"] or f["governance"].get("next_phase")!=a["required_next_phase"]: raise ProtocolError("F0i authorization drift")
    for name in ("gaussian_grouping","grounded_segment_anything"):
        s=c["sources"][name]; root=Path(s["path"])
        if _git_at(root,"rev-parse","HEAD")!=s["commit"] or _git_at(root,"rev-parse","HEAD^{tree}")!=s["tree"] or _git_at(root,"status","--porcelain"): raise ProtocolError(f"F0i source drift: {name}")
    for name,s in c["assets"].items(): _verify(Path(s["path"]),s["sha256"],name,int(s["bytes"]))
    s=c["sources"]["traced_file"]; _verify(Path(s["path"]),s["sha256"],"traced source",int(s["bytes"]))
    m=c["input_manifest"]; payload=json.loads(_verify(Path(m["path"]),m["sha256"],"input manifest",int(m["bytes"])).read_text())
    if payload.get("record_chain_sha256")!=m["record_chain_sha256"]: raise ProtocolError("F0i manifest drift")
    rows=[r for r in payload["records"] if r["scene"]==c["scene"]["name"]]
    expected=[(f,cam) for f in c["scene"]["frames"] for cam in c["scene"]["cameras"]]
    if len(rows)!=15 or [(int(r["frame"]),int(r["camera"])) for r in rows]!=expected: raise ProtocolError("F0i scene denominator/order drift")
    for r in rows: r["staging_filename"]=Path(r["path"]).name
    if c["execution"].get("pre_matmul_empty_cache") is not True or int(c["execution"]["attempt"]["sam_num_points_per_batch"])!=64: raise ProtocolError("F0i recovery drift")
    return c,rows

def run(config_path: Path, run_dir: Path) -> dict[str,Any]:
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"; c,rows=_validate_config(config_path)
    if run_dir.exists(): raise ProtocolError(f"refusing overwrite: {run_dir}")
    run_dir.mkdir(parents=True); _write_text(run_dir/"resolved_config.yaml",config_path.read_text()); ident=repository_source_identity()
    events=[{"event":"run_started","at_utc":_utc_now()}]; _write_jsonl(run_dir/"events.jsonl",events); _write_json(run_dir/"status.json",{"task_id":TASK_ID,"status":"running","source_commit":ident["commit"]})
    input_dir=run_dir/"artifacts/input"; input_dir.mkdir(parents=True)
    for r in rows: (input_dir/r["staging_filename"]).symlink_to(Path(r["path"]))
    c=dict(c); c["input_groups"]={"target":{"inputs":rows}}
    mon=ResourceMonitor(float(c["resources"]["monitor_interval_seconds"])); started=time.perf_counter(); mon.start()
    try:
        total=_nvidia_total_mib(); start=_nvidia_used_mib()
        if total!=24576 or start>int(c["resources"]["maximum_nvidia_at_start_mib"]): raise ProtocolError("F0i GPU start drift")
        attempt=_run_trace_attempt(c,c["execution"]["attempt"],input_dir,run_dir)
        if attempt["classification"]!="success" or len(attempt["masks"])!=15: raise ProtocolError("F0i output drift")
        pre=[e for e in attempt["trace"]["payload"]["events"] if e.get("event")=="pre_matmul"]
        if not pre or not all("empty_cache" in e and e["empty_cache"]["after"]["free_bytes"]>=e["empty_cache"]["before"]["free_bytes"] for e in pre): raise ProtocolError("F0i intervention evidence drift")
        mon.stop(); _write_jsonl(run_dir/"artifacts/resource_samples.jsonl",mon.samples); valid=[r for r in mon.samples if "monitor_error" not in r]
        res={"nvidia_total_mib":total,"nvidia_start_mib":start,"nvidia_peak_mib":max(int(r["gpu_used_mib"]) for r in valid),"cgroup_memory_peak_bytes":max(int(r["cgroup_memory_current_bytes"]) for r in valid),"sample_count":len(mon.samples),"monitor_error_count":len(mon.samples)-len(valid),"wall_seconds":time.perf_counter()-started,"disk_free_after_bytes":shutil.disk_usage(run_dir).free}; res["nvidia_headroom_mib"]=total-res["nvidia_peak_mib"]
        q=c["resources"]; checks={"nvidia_peak":res["nvidia_peak_mib"]<=int(q["maximum_nvidia_peak_mib"]),"nvidia_headroom":res["nvidia_headroom_mib"]>=int(q["required_nvidia_headroom_mib"]),"cgroup":res["cgroup_memory_peak_bytes"]<=int(q["maximum_cgroup_memory_bytes"]),"wall":res["wall_seconds"]<=float(q["maximum_wall_seconds"]),"disk":res["disk_free_after_bytes"]>=int(q["minimum_disk_free_bytes_after"]),"monitor":res["monitor_error_count"]==0}
        _write_json(run_dir/"artifacts/resources.json",res)
        if not all(checks.values()): raise ProtocolError(f"F0i resource gate: {checks}")
        summary={"task_id":TASK_ID,"status":"done","conclusion":c["decision"]["expected_conclusion"],"source_commit":ident["commit"],"source_tree":ident["tree"],"attempt":attempt,"empty_cache_call_count":len(pre),"resources":res,"resource_checks":checks,"quality_read":False,"full_materialization":False,"identity_training_authorized":False,"next_action":c["decision"]["next_action"],"m2_status":"pending","m3_status":"pending"}
        _write_json(run_dir/"summary.json",summary); events.append({"event":"run_completed","at_utc":_utc_now()}); _write_jsonl(run_dir/"events.jsonl",events); _write_json(run_dir/"manifest.json",{"task_id":TASK_ID,"status":"done","inventory":_inventory(run_dir)}); _write_json(run_dir/"status.json",{"task_id":TASK_ID,"status":"done","conclusion":summary["conclusion"],"source_commit":ident["commit"]}); return summary
    except BaseException as e:
        mon.stop(); _write_jsonl(run_dir/"artifacts/resource_samples.jsonl",mon.samples); events.append({"event":"run_blocked","at_utc":_utc_now(),"error":f"{type(e).__name__}: {e}"}); _write_jsonl(run_dir/"events.jsonl",events); _write_json(run_dir/"status.json",{"task_id":TASK_ID,"status":"blocked","error":f"{type(e).__name__}: {e}","source_commit":ident["commit"]}); raise

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=PROJECT/"configs/worldsim_v51/stage_f_f0i_scene1087_15_view_empty_cache_recovery_v1.yaml"); p.add_argument("--run-dir",type=Path,required=True); a=p.parse_args(); print(json.dumps(run(a.config.resolve(),a.run_dir.resolve()),ensure_ascii=False,sort_keys=True))
if __name__=="__main__": main()
