from pathlib import Path
import tarfile, json
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030')
patterns=[
 'dataset_r2/validation.json','dataset_r2/observations/manifest.json','dataset_r2/privileged/reference_manifest.json',
 'dataset_r2/observations/rgb/*.png','dataset_r2/privileged/references/*/*/rgb/*.png',
 'native_scene_r3/preparation_manifest.json',
 'timing_r1/returned.json','timing_r1/cuda_memory_peak.json',
 'smoke_r2/*.json','smoke_r2/*/returned.json','smoke_r2/*/restore_verification.json',
 'smoke_r2/render_entry/render_manifest.json',
 'smoke_r2/*/stop_trigger_timing.json','smoke_r2/*/cuda_memory_peak.json',
 'formal_r3_10k/*.json','formal_r3_10k/checkpoint_events/*.json',
 'formal_execution.json','formal_launch_contract.json','edit_execution.json',
 'edit_r3_10k/*.json','edit_r3_10k/**/*.png',
 'heldout_r3_10k/*.json','heldout_r3_10k/**/*.png',
 'EXPERIMENT_CONTRACT.json',
]
files=sorted(set(p for pattern in patterns for p in root.glob(pattern) if p.is_file()))
archive=root/'local_delivery.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in files:tar.add(p,arcname=p.relative_to(root).as_posix())
print(json.dumps({'files':len(files),'bytes':archive.stat().st_size,'path':str(archive)}))
