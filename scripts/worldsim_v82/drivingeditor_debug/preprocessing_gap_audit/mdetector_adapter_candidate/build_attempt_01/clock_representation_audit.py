"""Read-only source/clock arithmetic audit; no detector math or scan execution."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
CANDIDATE=ROOT.parent
DATA_AUDIT=CANDIDATE.parents[1]/"loss_audit/next_diagnostics_cpu/static_initialisation_actor_cpu/raw_field_availability_followup_cpu.json"
EXPECTED="5eae46e7023aafc578517642c830c87543fdc256d75cfef4912e2301ef38da2b"
sys.path.insert(0,str(CANDIDATE))
from generate_build_inputs import verify_vendor, read_waymo_yaml


def source_slice(path, start, end):
    lines=(CANDIDATE/"vendor"/path).read_text().splitlines()
    return {"path":path,"line_start":start,"line_end":end,"text":"\n".join(lines[start-1:end])}


def main():
    verify_vendor()
    raw=DATA_AUDIT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED: raise RuntimeError("frozen clock metadata hash mismatch")
    data=json.loads(raw)
    rows=data["all49_rows"]
    ticks=[int(r["timestamp_micros"]) for r in rows]
    origin=ticks[0]
    relative=[(t-origin)/1e6 for t in ticks]
    absolute=[t/1e6 for t in ticks]
    config=read_waymo_yaml(); dur=config["frame_dur"]
    gates={"fabs_pair_delta_lt_frame_dur":lambda d:abs(d)<dur,
           "fabs_pair_delta_gt_frame_dur":lambda d:abs(d)>dur,
           "buffer_delay_gate":lambda d:d>=config["buffer_delay"]-dur/2,
           "depth_map_duration_gate":lambda d:d>=config["depth_map_dur"]-dur/2,
           "positive_delta_gate":lambda d:d>0}
    flips=[]; maximum={"error_s":-1}
    for i,j in itertools.combinations(range(len(ticks)),2):
        a=absolute[j]-absolute[i]; r=relative[j]-relative[i]
        error=abs(a-r)
        if error>maximum["error_s"]:
            maximum={"earlier_frame":i,"later_frame":j,"absolute_delta_s":a,
                     "relative_delta_s":r,"error_s":error}
        for name,gate in gates.items():
            if gate(a)!=gate(r):
                flips.append({"earlier_frame":i,"later_frame":j,"gate":name,
                              "absolute_delta_s":a,"relative_delta_s":r,
                              "absolute_gate":gate(a),"relative_gate":gate(r)})
    means={"error_s":-1}
    for k,j,i in itertools.combinations(range(len(ticks)),3):
        a=(absolute[j]+absolute[i])/2-(absolute[k]+absolute[j])/2
        r=(relative[j]+relative[i])/2-(relative[k]+relative[j])/2
        if abs(a-r)>means["error_s"]:
            means={"older_frame":k,"middle_frame":j,"newer_frame":i,
                   "absolute_mean_interval_s":a,"relative_mean_interval_s":r,"error_s":abs(a-r)}
    example={"kind":"arithmetic counterexample, hypothetical exactly100000us interval, NOT a real scan",
             "epoch_start_us":origin,"epoch_end_us":origin+100000}
    a=(origin+100000)/1e6-origin/1e6; r=100000/1e6
    example.update({"absolute_delta_s":a,"relative_delta_s":r,
                    "absolute_lt_frame_dur":a<dur,"relative_lt_frame_dur":r<dur,
                    "absolute_gt_frame_dur":a>dur,"relative_gt_frame_dur":r>dur})
    evidence=[
        source_slice("include/m-detector/DynObjFilter.h",80,112),
        source_slice("include/m-detector/DynObjFilter.h",124,157),
        source_slice("include/m-detector/DynObjFilter.h",224,228),
        source_slice("include/m-detector/DynObjFilter.h",250,254),
        source_slice("include/m-detector/DynObjFilter.h",317,320),
        source_slice("include/m-detector/DynObjFilter.h",404,418),
        source_slice("src/DynObjFilter.cpp",258,275),
        source_slice("src/DynObjFilter.cpp",582,612),
        source_slice("src/DynObjFilter.cpp",994,1002),
        source_slice("src/DynObjFilter.cpp",1112,1146),
        source_slice("src/DynObjFilter.cpp",1170,1176),
        source_slice("src/DynObjFilter.cpp",1221,1225),
        source_slice("src/DynObjFilter.cpp",1255,1261),
        source_slice("src/DynObjFilter.cpp",1320,1327),
        source_slice("src/DynObjFilter.cpp",1353,1361),
        source_slice("src/DynObjFilter.cpp",1441,1448),
        source_slice("src/DynObjFilter.cpp",1494,1508),
        source_slice("src/DynObjFilter.cpp",1530,1563),
        source_slice("src/DynObjFilter.cpp",1584,1591),
        source_slice("src/DynObjFilter.cpp",1634,1638),
        source_slice("src/DynObjFilter.cpp",1674,1680),
        source_slice("src/DynObjFilter.cpp",1740,1746),
        source_slice("src/DynObjFilter.cpp",1768,1776),
        source_slice("src/DynObjCluster.cpp",31,45)]
    report={"scope":"Pinned source time-dataflow reading and IEEE754 clock arithmetic only",
        "source_revision":"0645dabe307a99f3b59a5a3d3007acfdcad1a39f",
        "raw_clock_receipt":str(DATA_AUDIT),"raw_clock_receipt_sha256":EXPECTED,
        "frames":len(ticks),"raw_first_scan_start_micros":origin,
        "first_relative_time_s":relative[0],"first_absolute_time_s":absolute[0],
        "source_findings":[
            "No direct absolute point/depth time comparison to0 or-1 found in the inspected core translation units.",
            "point_soph time defaults-1 at header87/108/131; active points get scan_end_time at cpp270 before cases and buffer insertion.",
            "Default DepthMap time0 is in its unused default constructor. Active maps are created with point.time at cpp589/610 and reset603.",
            "buffer_time0/last_point_pointernullptr are declared but not read by the core; buffer_dur is loaded but not otherwise used.",
            "Case2/3 local ti defaults0 at1114/1531, then is assigned a real time mean before the active time-chain comparisons. Waymo1 occluded_times_thr2=2 means the initial1-count early break does not skip the Case2 assignment.",
            "Case2/3 compare delta_t>0, not absolute time>0, at1325/1743; velocities use time differences and differences of timestamp means.",
            "Source warmup depends on an empty map list and elapsed buffer/map intervals, not the first timestamp being positive. This is source reasoning, not an execution proof.",
            "Header stamps are transport metadata; cluster fixed delta_t0.1 and ros::Time::now elapsed timers are independent of reference epoch."],
        "real_arithmetic_translation_invariant_for_active_history":True,
        "native_bitwise_or_branch_invariance_proven":False,
        "actual49_all1176_pair_max_double_delta_difference":maximum,
        "actual49_listed_time_gate_disagreements":flips,
        "actual49_chain_mean_interval_max_double_difference":means,
        "hypothetical_boundary_counterexample":example,
        "explicit_adapter_convention_requirement":"Caller must explicitly select and record frame-start absolute or segment-relative seconds, origin and receipt. Do not silently infer origin or assert physical Frame.pose timestamp/author producer parity. Origin0 can preserve epoch representation; first-start origin is a separate disclosed implementation choice.",
        "author_equivalence":False,"algorithm_labels_equivalence_tested":False,
        "detector_initialized":False,"real_scan_xyz_read":False,"gpu_used":False,
        "evidence":evidence}
    (ROOT/"clock_representation_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"frames":len(ticks),"actual_time_gate_disagreements":len(flips),
                     "max_pair_delta_error_s":maximum["error_s"],
                     "max_mean_interval_error_s":means["error_s"],"counterexample":example,
                     "no_direct_absolute_time_zero_gate_found":True,"no_detector_execution":True}))


if __name__=="__main__": main()
