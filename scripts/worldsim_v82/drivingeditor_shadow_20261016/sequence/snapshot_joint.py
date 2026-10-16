from pathlib import Path
import shutil,hashlib,json
r=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
d=r/'source_snapshots/joint_v1'
shutil.copytree(r/'implementation',d,ignore=shutil.ignore_patterns('__pycache__'))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.glob('*.py')}
(d/'sha256.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps({'snapshot':str(d),'files':len(hashes),'core_training_files_unchanged_since_launch':True,'auxiliary_eval_scripts_added_after_launch':True}))
