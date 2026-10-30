import json,tarfile,hashlib
from pathlib import Path
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/m1_assignment_r1")
for folder in r.iterdir():
 if folder.is_dir():
  for name in ("render_manifest.json","reference_metrics.json"):
   p=folder/name
   if p.exists():
    x=json.loads(p.read_text());x["routing"]="M1 frozen conditional static selection plus existing dynamic; diagnostic policy thresholds are not ideal ownership/GT";x["persistent_identity_solved"]=False;p.write_text(json.dumps(x,indent=2))
files=sorted(p for p in r.rglob("*") if p.is_file() and p.suffix in (".json",".png",".log") and "scene_loader" not in p.parts and p.name!="evidence_manifest.json")
manifest=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(r/"evidence_manifest.json").write_text(json.dumps(manifest,indent=2))
with tarfile.open(r/"evidence.tar.gz","w:gz") as t:
 for p in files+[r/"evidence_manifest.json"]:t.add(p,arcname=str(p.relative_to(r)))
print(len(files),sum(p.stat().st_size for p in files))
