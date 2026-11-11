from pathlib import Path
import json,hashlib,subprocess
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');O=C/'waymo_val065_delete30k_gpu_diagnosis_20261111_r1';P=C/'waymo_allframes_20261109_r1/public_api/drivingeditor_dynamic_edit'
old=json.loads((C/'waymo_val065_delete30k_diagnosis_20261109_r1/code_audit.json').read_text());baseline={x['file']:x['public_sha'] for x in old['files']}
names=['v44_cli.py','v44_factory.py','v44_scene.py','v44_calibrate.py','v44_gate_fit.py','v44_road.py','v44_manifest.py','ownership.py','scene_io.py']
rows=[]
for name in names:
    a=(P/name).read_bytes();b=(R/'scripts/worldsim_v82/drivingeditor_dynamic_edit'/name).read_bytes();sha=hashlib.sha256(a).hexdigest()
    rows.append(dict(file=name,public_matches_repository=a==b,public_matches_previous_execution_snapshot=sha==baseline[name]))
assert all(x['public_matches_repository'] and x['public_matches_previous_execution_snapshot'] for x in rows)
receipt=dict(repository_head_before_diagnosis_commit=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),runtime=str(P),rows=rows,algorithm_runtime_changed=False)
(O/'runtime_verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
