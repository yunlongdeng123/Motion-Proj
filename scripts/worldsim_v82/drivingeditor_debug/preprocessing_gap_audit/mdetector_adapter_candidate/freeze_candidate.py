"""Hash the reviewable candidate; excludes later dependency/build directories."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
excluded = {"__pycache__", "build_transport", "build_native", "build_attempt_01", "deps"}
files = []
for p in sorted(root.rglob("*")):
    if not p.is_file() or any(part in excluded for part in p.relative_to(root).parts): continue
    if p.name == "handoff_manifest.json" or p.name == "compat_probe": continue
    data = p.read_bytes()
    files.append({"path":p.relative_to(root).as_posix(),"bytes":len(data),
                  "sha256":hashlib.sha256(data).hexdigest()})
receipt = {"candidate":"CPU adapter contract/skeleton, no real detector execution",
           "source_revision":"0645dabe307a99f3b59a5a3d3007acfdcad1a39f",
           "vendor_files":20,"core_math_config_mutations":0,
           "detector_built_or_initialized":False,"real_detector_labels_available":False,
           "files":files}
out = root / "handoff_manifest.json"
out.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"files":len(files),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
                  "handoff_path":str(out)}))
