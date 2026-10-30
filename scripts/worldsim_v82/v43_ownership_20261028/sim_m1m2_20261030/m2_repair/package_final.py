import hashlib
import json
from pathlib import Path
import tarfile
root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/m2_repair_r1')
parity=json.loads((root/'depth_replay_parity.json').read_text())
assert len(parity)==6 and all(x['max_abs']['final']<=2e-6 for x in parity)
refits=json.loads((root/'local_refit_comparison.json').read_text())
assert len(refits)==2 and all(len(x['heldout_rows'])==4 for x in refits)
receipt=dict(primary_depth_corrected_outputs=[x['corrected'] for x in parity],
    original_RGB_assets_remain_valid='original main images C/alpha/E/final are bitwise identical, depth-replay parity receipt; use corrected NPZ for depth',
    local_refit=dict(arms=['refit_assignment','refit_split'],updates_per_arm=200,total_updates=400,trainable='exact M1 selected DC+opacity, plus split background child DC+opacity in M2; higher SH/geometry/nonselected/dynamic/environment frozen',loss='full-frame original trainRGB L1, no masks, no edited/heldout RGB',lr=dict(DC=.0025,opacity=.05),optimizer='Adam eps1e-8',single_frame=8,save_reload_parity=[x['local_refit']['save_reload_render_max_abs'] for x in refits],capacity_and_trainable_count=[dict(arm=x['name'],rows=x['local_refit']['trainable_rows'],parameters=x['local_refit']['trainable_parameters']) for x in refits]),
    depth_bug='none features originally used pre-repair z; corrected features regenerated using modified means3D before native raster, 48 forward renders replayed, C/alpha/E/final parity receipt',
    native_checkpoint_unchanged=True,GT_RGB_replaced=False,mathematical_upper_bound=False,quality_accepted=False,human_verdict=None)
(root/'final_completion_receipt.json').write_text(json.dumps(receipt,indent=2))
files=[root/x for x in ['depth_replay_parity.json','depth_replay_queue_receipt.json','local_refit_comparison.json','final_completion_receipt.json']]
for p in root.iterdir():
    if p.is_dir() and (p.name.endswith('_depth_r2') or p.name in ['refit_assignment','refit_split','heldout_refit_assignment','heldout_refit_split']):
        files.extend(f for f in p.rglob('*') if f.is_file() and (f.suffix=='.json' or (p.name in ['refit_assignment','refit_split','heldout_refit_assignment','heldout_refit_split'] and f.suffix=='.png' and any(f.name.endswith('_'+k+'.png') for k in ['final','white','alpha']))))
manifest=[dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(root/'final_evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'final_evidence.tar.gz','w:gz') as tar:
    for p in files+[root/'final_evidence_manifest.json']:tar.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(files=len(files),bytes=(root/'final_evidence.tar.gz').stat().st_size)))
