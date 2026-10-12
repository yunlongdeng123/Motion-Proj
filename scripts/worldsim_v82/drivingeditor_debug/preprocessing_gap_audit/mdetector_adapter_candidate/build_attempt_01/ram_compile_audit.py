"""Exact sizeof via an unlinked ELF object; memory state read only, no init."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CANDIDATE=ROOT.parent
PREFIX=CANDIDATE/"deps/prefix/usr"
sys.path.insert(0,str(CANDIDATE))
from generate_build_inputs import verify_vendor, read_waymo_yaml


def run(name,args):
    p=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=60)
    (ROOT/(name+".log")).write_text(p.stdout+p.stderr)
    if p.returncode: raise RuntimeError(p.stdout+p.stderr)
    return {"command":args,"exit_code":p.returncode,"log":name+".log"}


def read_memory():
    meminfo=Path("/proc/meminfo").read_text()
    host={}
    for line in meminfo.splitlines():
        parts=line.split()
        if parts: host[parts[0].rstrip(":")]=int(parts[1])*1024
    cgroup={}
    for name in ("memory.max","memory.current","memory.high","memory.peak","memory.stat","memory.events"):
        p=Path("/sys/fs/cgroup")/name
        if p.exists(): cgroup[name]=p.read_text().strip()
    stat={k:int(v) for k,v in (x.split() for x in cgroup.get("memory.stat","").splitlines())}
    maximum=cgroup.get("memory.max","max")
    current=int(cgroup.get("memory.current","0"))
    hard_limit=None if maximum=="max" else int(maximum)
    hard_headroom=None if hard_limit is None else max(0,hard_limit-current)
    effective=min(host["MemAvailable"],hard_headroom) if hard_headroom is not None else host["MemAvailable"]
    return {"utc":datetime.now(timezone.utc).isoformat(),
            "host_mem_available_bytes":host["MemAvailable"],"host_mem_total_bytes":host["MemTotal"],
            "cgroup_raw":cgroup,"cgroup_stat":stat,"cgroup_hard_limit_bytes":hard_limit,
            "cgroup_current_bytes":current,"cgroup_unoccupied_hard_headroom_bytes":hard_headroom,
            "effective_unoccupied_headroom_bytes":effective,
            "potentially_reclaimable_file_cache_observation_bytes":stat.get("inactive_file",0),
            "cache_is_not_counted_as_automatically_available":True}


def main():
    verify_vendor()
    compile_command=["g++","-std=c++17","-O3","-fopenmp","-c",str(ROOT/"abi_layout.cpp"),
        "-include",str(CANDIDATE/"native/prelude.hpp")]
    for include in (CANDIDATE/"generated/transport_include",CANDIDATE/"native",
                    CANDIDATE/"vendor/include",PREFIX/"include",PREFIX/"include/eigen3",
                    PREFIX/"include/pcl-1.12"):
        compile_command.extend(["-I",str(include)])
    compile_command.extend(["-o",str(ROOT/"abi_layout.o")])
    compile_result=run("abi_compile",compile_command)
    extract_result=run("abi_extract",["objcopy","--dump-section",
        ".mdetector_layout="+str(ROOT/"abi_layout.bin"),str(ROOT/"abi_layout.o")])
    raw=(ROOT/"abi_layout.bin").read_bytes()
    fields=("magic","version","point_soph_size","point_soph_align","Point_Cloud_size",
            "Point_Cloud_align","DepthMap_size","DepthMap_align","DynObjFilter_size",
            "DynObjCluster_size","PCL_PointType_size","PCL_PointType_align",
            "point_pointer_size","vector_point_pointer_size","float_size","int_size","double_size")
    if len(raw)!=8*len(fields): raise RuntimeError("unexpected ELF layout payload")
    layout=dict(zip(fields,struct.unpack("<"+"Q"*len(fields),raw)))
    if layout["magic"]!=0x4d44455441424931 or layout["version"]!=1:
        raise RuntimeError("wrong ELF layout tag")
    config=read_waymo_yaml(); n=config["points_num_perframe"]
    xy=math.ceil(200.0/config["cluster_Voxel_revolusion"])
    z=math.ceil(40.0/config["cluster_Voxel_revolusion"])
    grid=xy*xy*z
    slots=math.floor((config["max_depth_map_num"]*config["depth_map_dur"]+config["buffer_delay"])/config["frame_dur"]+0.5)+1
    depth_pixels=564393 # unchanged MAX_2D_N from the pinned header
    pos_count=(2*1257+1)*(2*449+1) # unchanged hor_num/ver_num source initialization
    components={
        "three_dense_voxel_grid_payload":3*grid*layout["Point_Cloud_size"],
        "fixed_point_soph_ring_payload":slots*n*layout["point_soph_size"],
        "point_soph_buffer_pointer_payload":config["buffer_size"]*layout["point_pointer_size"],
        "pos_offset_live_element_payload":pos_count*layout["int_size"],
        "first_history_cloud_reserved_payload":400000*layout["PCL_PointType_size"],
        "five_depth_maps_fixed_outer_vectors_arrays":config["max_depth_map_num"]*depth_pixels*(
             layout["vector_point_pointer_size"]+4*layout["float_size"]+3*layout["int_size"]),
        "five_depth_map_metadata":config["max_depth_map_num"]*layout["DepthMap_size"],
    }
    # This next group is the upper configured per-frame payload for specifically
    # observed reserve(n) arrays/clouds, not the entire algorithm's peak allocation.
    per_frame={
        "six_timing_arrays_at_configured_capacity":6*n*layout["double_size"],
        "point_index_and_point_pointer_arrays":n*(layout["int_size"]+layout["point_pointer_size"]),
        "five_result_cloud_reserves_plus_raw_cloud":6*n*layout["PCL_PointType_size"],
        "adapter_native_input_cloud_reserve":n*layout["PCL_PointType_size"],
    }
    memory=read_memory()
    known=sum(components.values()); frame_payload=sum(per_frame.values())
    # Proposed budget is an adapter execution policy, never an author threshold.
    budget=max(12*1024**3,math.ceil((known+frame_payload)*1.35/(1024**3))*1024**3)
    reserve=8*1024**3
    report={"scope":"Compile-only ABI payload arithmetic and current cgroup/host memory read",
            "source_revision":"0645dabe307a99f3b59a5a3d3007acfdcad1a39f",
            "core_source_config_mutations":0,"layout":layout,
            "compile":compile_result,"elf_metadata_extract":extract_result,
            "abi_layout_bin_sha256":hashlib.sha256(raw).hexdigest(),
            "grid_edge_xy":xy,"grid_edge_z":z,"grid_elements_per_array":grid,
            "point_soph_ring_slots":slots,"known_main_payload_bytes":components,
            "known_main_payload_sum_bytes":known,"configured_per_frame_known_payload_bytes":per_frame,
            "known_main_plus_listed_per_frame_payload_bytes":known+frame_payload,
            "not_statically_quantified":[
                "std::vector capacity growth/allocator headers/alignment/fragmentation and control blocks",
                "per-depth-pixel pointer-vector capacities and populated depth-map payload history",
                "cluster temporary and event/ground/OOBB/region vectors, copied clouds and voxel payloads",
                "adapter raw-lineage/input/output transport representation and future Python binding arrays",
                "PCL/FLANN/Eigen/TBB scratch memory, worker stacks, page rounding and retained allocations",
                "runtime peak RSS/page-cache reclaim behavior; detector has never been initialized"],
            "memory_observation":memory,"proposed_unreviewed_peak_budget_bytes":budget,
            "proposed_shared_cgroup_reserve_bytes":reserve,
            "required_unoccupied_headroom_bytes":budget+reserve,
            "headroom_comparison_passes_proposed_budget_only":memory["effective_unoccupied_headroom_bytes"]>=budget+reserve,
            "execution_authorized":False,"backend_or_filter_constructed":False,
            "detector_initialized":False,"real_scans_processed":False,"gpu_used":False,
            "claim":"Arithmetic is exact for listed ABI payload terms, not a measured or proven peak RSS."}
    (ROOT/"ram_compile_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    verify_vendor()
    print(json.dumps({"layout":layout,"known_payload_gib":known/1024**3,
        "known_plus_listed_frame_gib":(known+frame_payload)/1024**3,
        "cgroup_unoccupied_headroom_gib":memory["cgroup_unoccupied_hard_headroom_bytes"]/1024**3,
        "proposed_peak_budget_gib":budget/1024**3,"required_with_reserve_gib":(budget+reserve)/1024**3,
        "comparison_pass":report["headroom_comparison_passes_proposed_budget_only"],
        "no_construction_or_execution":True}),flush=True)


if __name__=="__main__": main()
