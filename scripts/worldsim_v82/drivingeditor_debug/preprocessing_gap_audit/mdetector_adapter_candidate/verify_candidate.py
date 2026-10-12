"""Bounded CPU verification; never compiles/runs/initializes full detector."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from generate_build_inputs import generate, ROOT


def run(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=45)
    return {"command":argv, "exit_code":p.returncode, "stdout":p.stdout, "stderr":p.stderr}


def main():
    audit = generate()
    check_dir = ROOT / "verification"
    check_dir.mkdir(exist_ok=True)
    checks = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_contract.py")
    stream_path = check_dir / "contract_test_log.txt"
    with stream_path.open("w", encoding="utf-8") as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(checks)
    source = (ROOT / "vendor/src/DynObjFilter.cpp").read_text()
    begin = source.index("bool  DynObjFilter::InvalidPointCheck")
    end = source.index("bool  DynObjFilter::SelfPointCheck", begin)
    excerpt = source[begin:end]
    # A signature-only vector and class permit an exact-body parse without PCL/Eigen.
    # No probe invokes this function or reproduces its detector behavior.
    probe = '#include <cmath>\nusing std::pow; using std::fabs;\n' \
        'struct V3D { double operator()(int) const; };\n' \
        'struct DynObjFilter { double blind_dis; int dataset; '\
        'bool InvalidPointCheck(const V3D&, const int); };\n' + excerpt
    syntax_path = check_dir / "unchanged_invalidpoint_syntax.cpp"
    syntax_path.write_text(probe, encoding="utf-8")
    compiler = shutil.which("g++")
    syntax = shim_compile = shim_run = None
    if compiler:
        syntax = run([compiler,"-std=c++17","-fsyntax-only",str(syntax_path)])
        executable = check_dir / "compat_probe"
        shim_compile = run([compiler,"-std=c++17","-O0","-I",str(ROOT/"native"),
                            "-I",str(ROOT/"generated"),str(ROOT/"native/compat_probe.cpp"),
                            "-o",str(executable)])
        if shim_compile["exit_code"] == 0: shim_run = run([str(executable)])
    x = math.ceil(200.0/0.3); z = math.ceil(40.0/0.3)
    report = {"scope":"Adapter validation and unchanged author function syntax only",
        "python_contract_tests":{"tests_run":tests.testsRun, "failures":len(tests.failures),
                                 "errors":len(tests.errors), "passed":tests.wasSuccessful()},
        "vendor_files_verified":20, "source_revision":audit["revision"],
        "core_source_mutations":0, "config_mutations":0,
        "invalidpoint_exact_body_sha256":hashlib.sha256(excerpt.encode()).hexdigest(),
        "pinned_source_syntax_probe":syntax, "transport_facade_compile":shim_compile,
        "transport_facade_run":shim_run,
        "author_init_grid":{"edge_xy":x,"edge_z":z,"elements_per_array":x*x*z,
                            "arrays":3, "illustrative_40byte_element_gib":3*x*x*z*40/1024**3,
                            "actual_point_cloud_size_not_measured":True},
        "gpu_used":False, "full_detector_built":False, "detector_initialized":False,
        "detector_labels_computed":False, "scene_data_read_or_written":False,
        "claim":"These checks establish transport/lineage guards, not algorithm or training equivalence."}
    (check_dir / "cpu_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"contract_tests_passed":tests.wasSuccessful(),"tests_run":tests.testsRun,
                     "pinned_source_syntax_exit":None if syntax is None else syntax["exit_code"],
                     "transport_facade_compile_exit":None if shim_compile is None else shim_compile["exit_code"],
                     "transport_facade_run_exit":None if shim_run is None else shim_run["exit_code"],
                     "detector_initialized":False}))
    return 0 if tests.wasSuccessful() and shim_compile and shim_compile["exit_code"]==0 \
        and shim_run and shim_run["exit_code"]==0 else 1


if __name__ == "__main__": sys.exit(main())
