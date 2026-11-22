"""Preserve native renderer sources and the canonical field definition."""
from pathlib import Path
import hashlib,json,shutil

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
cfg=json.loads((ROOT/'baseline_contract.json').read_text())
repo=Path(cfg['repo']);base=repo/'scripts/worldsim_v82/drivingeditor_dynamic_edit'
out=ROOT/'retained_runtime_snapshot';assert (out/'receipt.json').is_file()
rows=[]
sources=[p for p in sorted((base/'native').rglob('*')) if p.is_file() and p.suffix in {'.cu','.cuh','.cpp','.h','.hpp','.c','.cmake'} and not {'build','__pycache__'}.intersection(p.parts)]
sources.append(Path(cfg['actor_field_module']))
for path in sources:
    if path==Path(cfg['actor_field_module']):dst=out/'canonical_field_definition.py'
    else:dst=out/'drivingeditor_dynamic_edit'/path.relative_to(base)
    assert not dst.exists();dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    rows.append(dict(source=str(path),relative_snapshot=dst.relative_to(out).as_posix(),bytes=dst.stat().st_size,
                     sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
receipt=dict(schema='R8.native_extension_source_snapshot.v1',complete=True,files=rows,GPU_calls=0,shared_checkout_modified=False,
    replay_roundoff_evidence='native/surface_rasterizer/cuda_rasterizer/backward.cu uses atomicAdd for dL_dcolors accumulation; post-restore update may be float32-close without bitwise identity')
(out/'native_extension_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows))))
