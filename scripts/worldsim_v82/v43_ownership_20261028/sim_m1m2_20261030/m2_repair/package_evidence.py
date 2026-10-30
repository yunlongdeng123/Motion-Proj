import hashlib
import json
from pathlib import Path
import tarfile

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/m2_repair_r1')
runs=['strict_position_r2','strict_position_scale','balanced_position','balanced_position_scale','balanced_position_scale_split_r2','heldout_balanced_split']
method=dict(method='M2 conditional truth-assisted support-centroid position / scale cap / controlled mixed split',
    m1_candidates_reused=True,candidate_indices_verified='comparison.json, exact equal for all 3 keys and all five repair arms',
    modifiable=['selected static means3D','selected static scales for scale arm','selected mixed opacity and one extra child for split arm'],
    unchanged_params=['learned dynamic branch','nonselected static rows','SH/color','rotations','environment','checkpoint files'],
    optimization_updates=0,privileged_trainview_geometry_used_for_repair=True,GT_RGB_replaced=False,
    position_target='mean of accepted M1 independent first-hit vehicle support positions; not exact mesh projection',
    scale_cap_world_m=.08,split_trigger='balanced selected AND mixed finite-support AND actual original alphaT to body and nonvehicle each >1e-4',
    opacity_conservation='optical thickness fractions, exact transmittance only at co-located equal footprints; relocation changes pixelwise alpha',
    heldout='load fixed frame8 repaired geometry from train query, never use heldout geometry for selection or repair; score only none against available none reference',
    inference_scope='query-conditioned diagnostic, not a persistent camera-independent Gaussian model or deployable predictor',
    mathematical_upper_bound=False,quality_accepted=False,
    failures=[dict(folder='strict_position',reason='position-repaired none incorrectly compared to unmodified native none; assertion removed, all output rerun in strict_position_r2'),dict(folder='balanced_position_scale_split',reason='float64 depth versus float32 calibrated camera; cast to float32, rerun new folder')])
(root/'method_receipt.json').write_text(json.dumps(method,indent=2))
for name in runs:
    folder=root/name
    for fn in ['render_manifest.json','reference_metrics.json']:
        path=folder/fn
        if path.exists():
            d=json.loads(path.read_text())
            if 'routing' in d:
                d['routing']='fixed M1 selected static rows repaired and edited with unchanged learned dynamic branch; split background child remains static'
            d['method_receipt']=str(root/'method_receipt.json')
            path.write_text(json.dumps(d,indent=2))
files=[root/'comparison.json',root/'queue_receipt.json',root/'method_receipt.json']
for name in runs:
    files.extend(p for p in (root/name).rglob('*') if p.is_file() and p.suffix in ('.json','.png'))
manifest=[dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(root/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'evidence.tar.gz','w:gz') as tar:
    for path in files+[root/'evidence_manifest.json']:
        tar.add(path,arcname=str(path.relative_to(root)))
print(json.dumps(dict(files=len(files),archive=str(root/'evidence.tar.gz'),bytes=(root/'evidence.tar.gz').stat().st_size)))
