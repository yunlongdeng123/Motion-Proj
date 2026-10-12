"""Single-worker configure/compile receipt. No detector/library execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CANDIDATE=ROOT.parent
sys.path.insert(0,str(CANDIDATE))
from generate_build_inputs import verify_vendor


def step(name,args):
    log=ROOT/(name+".log")
    start=time.time()
    with log.open("w") as f:
        p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,timeout=600)
    item={"step":name,"command":args,"exit_code":p.returncode,"elapsed_s":time.time()-start,
          "log":log.name,"log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()}
    print(json.dumps(item),flush=True)
    print(log.read_text()[-10000:],flush=True)
    return item


def main():
    verify_vendor()
    free=shutil.disk_usage(CANDIDATE).free
    if free<30*1024**3: raise RuntimeError("free disk under30GiB")
    report={"scope":"Native core and adapter compilation only, no construction/execution",
            "source_revision":"0645dabe307a99f3b59a5a3d3007acfdcad1a39f",
            "vendor_files_verified":20,"vendor_source_config_mutations":0,
            "stock_candidate_configure":{"exit_code":1,
                "diagnostic":"PCLConfig requires Boost system/filesystem/date_time/iostreams/serialization binaries; no core compiled in that probe",
                "evidence":"stock_cmake_probe/CMakeCache.txt"},
            "build_portability_change":"Direct real private PCL headers and shared libraries, bypassing distribution PCLConfig's unused Boost binary import requirements",
            "steps":[],"gpu_used":False,"detector_initialized":False,
            "real_scans_processed":False,"real_detector_labels_computed":False}
    prefix=CANDIDATE/"deps/prefix/usr"
    config=step("native_configure",["cmake","-S",str(ROOT),"-B",str(ROOT/"build"),
        "-DCMAKE_BUILD_TYPE=Release","-DCMAKE_PREFIX_PATH="+str(prefix),
        "-DBOOST_INCLUDEDIR="+str(prefix/"include")])
    report["steps"].append(config)
    if config["exit_code"]==0:
        report["steps"].append(step("native_compile",["cmake","--build",str(ROOT/"build"),
            "--target","mdetector_candidate","-j1"]))
    archive=ROOT/"build/libmdetector_candidate.a"
    report["full_native_core_and_adapter_compilation_passed"]=len(report["steps"])==2 and all(s["exit_code"]==0 for s in report["steps"])
    if report["full_native_core_and_adapter_compilation_passed"] and archive.exists():
        report["archive"]={"path":str(archive),"bytes":archive.stat().st_size,
                           "sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}
    verify_vendor()
    report["vendor_files_reverified_after_build"]=20
    report["free_bytes_after"]=shutil.disk_usage(CANDIDATE).free
    report["attempt_plus_prefix_bytes_after"]=sum(p.lstat().st_size for top in (ROOT,CANDIDATE/"deps/prefix")
        for p in top.rglob("*") if p.is_file() or p.is_symlink())
    if report["attempt_plus_prefix_bytes_after"]>8*1024**3:
        report["resource_limit_exceeded"]=True
    (ROOT/"native_build_receipt.json").write_text(json.dumps(report,indent=2)+"\n")
    return 0 if report["full_native_core_and_adapter_compilation_passed"] else 1


if __name__=="__main__": sys.exit(main())
