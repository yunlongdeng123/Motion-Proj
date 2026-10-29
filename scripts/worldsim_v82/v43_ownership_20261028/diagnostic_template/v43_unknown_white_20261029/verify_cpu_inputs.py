"""CPU input audit; compares roles against the actual JavaScript classifier.

This deliberately does not claim CUDA rendering, alpha identity, visual quality
or checkpoint loading have been tested. Those gates execute in the GPU runner.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location("unknown_white_runner", HERE / "render_unknown_white.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
contract, evidence, bundle = runner.load_bundle(HERE / "render_inputs/input_contract.json")
assert "torch" not in sys.modules

javascript = r"""
const fs=require('node:fs'),crypto=require('node:crypto');
const source=fs.readFileSync(process.argv[1],'utf8');
const raw=fs.readFileSync(process.argv[2],'utf8');
const E=JSON.parse(raw.slice(raw.indexOf('=')+1).trim().replace(/;$/,''));
const begin=source.indexOf('function actorPartition(){');
const end=source.indexOf('function rowRole(',begin);
if(begin<0||end<0)throw Error('Actual runtime classifier not found');
const S={raw:77},partitions=new Map();
function evidence(){return E.frames[String(S.raw)];}
function parameters(){return {radius:5,fraction:.2};}
const classify=eval('('+source.slice(begin,end).trim()+')');
const results={};
for(const raw of E.raws){S.raw=raw;const result=classify();
 const ids=result.roles.flatMap((r,i)=>r.role==='unknown'?[i]:[]);
 results[raw]={unknown_sha256:crypto.createHash('sha256').update(JSON.stringify(ids)).digest('hex'),counts:result.counts};}
process.stdout.write(JSON.stringify(results));
"""
runtime = ROOT / "work/v43_r2_regression_20261029/r2_ownership_runtime.js"
source_roles = bundle / contract["source_assets"]["ownership"]["path"]
actual = json.loads(subprocess.check_output(["node", "--eval", javascript, str(runtime), str(source_roles)], text=True))
comparisons = []
for raw in runner.RAWS:
    unknown, counts = runner.actor_partition(evidence["frames"][str(raw)], evidence["radii"])
    digest = hashlib.sha256(json.dumps(unknown, separators=(",", ":")).encode()).hexdigest()
    assert digest == actual[str(raw)]["unknown_sha256"]
    assert counts == actual[str(raw)]["counts"]
    comparisons.append(dict(raw=raw, actor_unknown=len(unknown), exact_js_role_match=True,
                            static_unknown=len(evidence["frames"][str(raw)]["static"]["final_removed_ids"])))

from PIL import Image
image_count = 0
for gid, group in contract["groups"].items():
    for raw, entry in group["frames"].items():
        with Image.open(bundle / entry["baseline_file"]) as image:
            assert image.size == (480, 320) and image.mode == "RGB", (gid, raw, image.size, image.mode)
            image.load()
        image_count += 1
assert image_count == 204
prepared = runner.read(bundle / "render_manifest.prepared.json")
assert prepared["status"] == "prepared" and not prepared["gpu_executed"]
assert prepared["verified_group_frame_count"] == 0
assert all(not frame["images"] and frame["status"] == "prepared"
           for group in prepared["groups"].values() for frame in group["frames"].values())
assert all(prepared["groups"][gid]["status"] == "evidence_needed" for gid in runner.EVIDENCE_NEEDED)
assert "torch" not in sys.modules
report = dict(schema="unknown-white-cpu-input-audit-v1", passed=True, gpu_executed=False,
              torch_imported=False, actor_frames_exactly_match_actual_js=49,
              source_baseline_pngs_verified=image_count, source_evidence_and_RGB_sha_verified=True,
              prepared_manifest_contains_no_render_images=True,
              unsupported_groups_explicitly_need_evidence=list(runner.EVIDENCE_NEEDED),
              runtime_classifier_sha256=runner.sha(runtime),
              input_contract_sha256=runner.sha(bundle / "input_contract.json"), records=comparisons,
              remaining_gpu_checks=["checkpoint/state binding", "native baseline saved-PNG equality",
                                    "alpha/radii/depth identity", "white SH native raster output",
                                    "source model/env unchanged", "visual inspection"])
runner.write(HERE / "cpu_input_audit.json", report)
print(json.dumps({k: v for k, v in report.items() if k not in ("records", "remaining_gpu_checks")}, indent=2))
