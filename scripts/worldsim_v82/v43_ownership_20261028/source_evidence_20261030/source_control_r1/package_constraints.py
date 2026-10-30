from pathlib import Path
import json,hashlib,tarfile
import diff_gaussian_rasterization as raster
r=Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/source_control_r1");out=r/"constraint_audit_r2"
refs=json.loads((out/"source_evidence.json").read_text())
for path in ("/root/autodl-tmp/drivingeditor_repro/deps/diff-gaussian-rasterization/cuda_rasterizer/forward.cu","/root/autodl-tmp/drivingeditor_repro/infra/rasterizer_compact/cuda_rasterizer/forward.cu"):
 p=Path(path);lines=p.read_text().splitlines();refs.append(dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),excerpts=[dict(start=18,end=78,text="\n".join(str(i+1)+": "+lines[i] for i in range(17,78)))]))
(out/"source_evidence.json").write_text(json.dumps(refs,indent=2))
module_path=Path(raster.__file__);binary=Path(raster._C.__file__)
(out/"active_renderer_module.json").write_text(json.dumps(dict(python_module=str(module_path),python_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),binary=str(binary),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),cuda_sources="candidate source copies have matching positive-only SH clamp; active binary source-build provenance not reconstructed; independent native/precomputed color parity and native HDR provide direct runtime corroboration"),indent=2))
files=list(out.glob("*"))+[r/"code/audit_constraints.py",r/"code/generic_source_audit.py",r/"code/render_generic_source.py",r/"code/generic_source_config.json"]
manifest=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files];(r/"constraints_delivery_manifest.json").write_text(json.dumps(manifest,indent=2))
with tarfile.open(r/"constraints_delivery.tar.gz","w:gz") as t:
 for p in files+[r/"constraints_delivery_manifest.json"]:t.add(p,arcname=str(p.relative_to(r)))
print(len(files),sum(p.stat().st_size for p in files))
