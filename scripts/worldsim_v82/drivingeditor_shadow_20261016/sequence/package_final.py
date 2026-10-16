"""Batch collection of owned immutable evidence; avoids repeated SSH setup."""
import json,zipfile,argparse
from pathlib import Path
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_sequence_20261016_v2')
p=argparse.ArgumentParser();p.add_argument('--latest',action='store_true');args=p.parse_args()
train='unselected_clean_v1' if args.latest else 'contribution_frozen_v1';evaluation='final49_unselected_v1' if args.latest else 'final49_contribution_v1'
assert json.loads((root/evaluation/'status.json').read_text())['state']=='complete'
dest=root/('final_delivery_package_v2.zip' if args.latest else 'final_delivery_package_v1.zip')
if dest.exists():raise SystemExit('Preserve existing package')
files=[]
def add(path,arc=None):
    path=Path(path)
    if path.is_file():files.append((path,arc or path.relative_to(root).as_posix()))
for p in (root/train).glob('*.json'):add(p)
for name in ['final.pt','training.jsonl']:add(root/train/name)
for p in (root/train/'previews').rglob('*.png'):add(p)
for p in (root/evaluation).glob('*.json'):add(p)
for p in (root/evaluation).glob('*.mp4'):add(p,'raw_videos/'+evaluation+'/'+p.name)
for p in (root/evaluation/'comparison').glob('*.png'):add(p,'reviews/'+evaluation+'/comparison_frames/'+p.name)
for stream in ['observed','static','delete_target','translate_smooth','yaw_smooth_15','yaw_smooth_30','fixed_static','fixed_translate','confidence']:
    for fid in ['0008089','0008101','0008113','0008125','0008137']:
        add(root/evaluation/stream/(fid+'.png'),'reviews/'+evaluation+'/'+stream+'/'+fid+'.png')
for directory in ['guard_refinement_v2','contribution_guarded_audit','unselected_foreground_audit']:
    for p in (root/directory).glob('*'):
        if p.suffix in ['.json','.pt','.npz','.png']:add(p)
for name in ['training_accounting.json','new_visibility_audit.json','body_endpoint_audit.json','away_reference_audit.json','shadow_math_validation.json']:
    add(root/name,'evidence/'+name)
for name in [train+'_launch.json','unselected_follow_launch.json' if args.latest else 'contribution_follow_launch.json']:
    add(root/name,'evidence/'+name)
add(root/'source_snapshots'/train/'sha256.json',train+'/source_sha256.json')
with zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_STORED) as archive:
    for path,arc in files:
        assert path.resolve().is_relative_to(root.resolve());archive.write(path,arc)
print(json.dumps({'package':str(dest),'files':len(files),'bytes':dest.stat().st_size}))
