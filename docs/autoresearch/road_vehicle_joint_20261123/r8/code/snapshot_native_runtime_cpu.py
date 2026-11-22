"""Read-only copy of the exact shared SDK source; never stage its dirty files."""
from pathlib import Path
import hashlib, json, shutil, subprocess

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
source=REPO/'scripts/worldsim_v82/drivingeditor_dynamic_edit'
out=ROOT/'retained_runtime_snapshot';out.mkdir(exist_ok=False)
rows=[]
for path in sorted(source.rglob('*.py')):
    if '__pycache__' in path.parts:continue
    dst=out/'drivingeditor_dynamic_edit'/path.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dst)
    data=path.read_bytes();assert dst.read_bytes()==data
    rows.append(dict(source=str(path),relative_snapshot=dst.relative_to(out).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain=v1','--untracked-files=all'])
(out/'receipt.json').write_text(json.dumps(dict(schema='R8.native_runtime_source_snapshot.v1',complete=True,HEAD=head,
    source_files=rows,source_checkout_porcelain_SHA256=hashlib.sha256(status).hexdigest(),
    imported_by_training=False,shared_checkout_modified=False,shared_dirty_files_staged=False,GPU_calls=0),indent=2)+'\n')
print(json.dumps(dict(snapshot=str(out),files=len(rows),bytes=sum(x['bytes'] for x in rows))))
