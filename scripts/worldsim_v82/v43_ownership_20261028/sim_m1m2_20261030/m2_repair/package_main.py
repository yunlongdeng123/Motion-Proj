import hashlib
import json
from pathlib import Path
import tarfile

root=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/m2_repair_r1')
runs=['recall_position','recall_position_scale','recall_position_scale_split']
comparison=json.loads((root/'recall_comparison.json').read_text())
assert all(c['indices_identical'] for run in comparison for c in run['selection_checks'])
method=dict(main_comparison=runs,pilot_appendix=['strict_position_r2','strict_position_scale','balanced_position','balanced_position_scale','balanced_position_scale_split_r2'],
    method='truth-assisted, fixed high-recall M1 candidates, actual native alphaT-weighted first-hit XYZ moment repair',
    candidate_source='m1_assignment_r1/recall_verified selected exact match; moment pass only supplies target positions/valid',
    split_weights='original recall_verified body_alphaT/background_alphaT; threshold1e-4 each and finite moment targets; no depth gate',
    position_rule='replace means with query-conditioned body first-hit XYZ moment, including wrong-depth contributions',
    scale_rule='cap maximum scale at .08 world metres and preserve anisotropy; fixed pilot rule, no threshold search',
    split_rule='one parent to vehicle+static child at body/background moments; inherit SH/quaternion/capped scale; alpha optical-depth split by body/(body+background) clipped [.01,.99]',
    conservation='co-located equal-footprint transmittance only; pixelwise alpha changes after relocation',
    modifiable=['selected static positions','selected static scales in scale arms','mixed selected opacity and extra child in split arm'],
    frozen=['original learned dynamic','nonselected static','SH/color','quaternions','environment','checkpoint weights'],
    optimization_updates=0,RGB_fitting=False,GT_RGB_replacement=False,
    privileged_reference_geometry_used_for_repair=True,mathematical_upper_bound=False,deployable_claim=False,
    heldout='exact frame8 repaired geometry NPZ replay under new query camera for all four edits; independent heldout references read only for scoring',
    missing_moment_selected_handling='leave parameters unchanged, preserve original edit membership',
    interpretation='negative geometry repair probe; does not establish a general representation repair limit, and cannot remove the need to fit appropriate appearance/covariance',
    operational_failures='pilot strict_position native-none assertion inappropriate for changed none; pilot split dtype mismatch; each preserved and replayed in a new directory',
    actual_render_count_main=48,quality_accepted=False,human_verdict=None)
validation=[]
for name in runs:
    tr=root/name; ho=root/('heldout_'+name)
    manifests=[json.loads((tr/'render_manifest.json').read_text()),json.loads((ho/'render_manifest.json').read_text())]
    assert [len(x['frames']) for x in manifests]==[12,4]
    assert all(x['original_checkpoint_unchanged'] for x in manifests)
    repair=json.loads((tr/'responsibility_summary.json').read_text())
    repair['policy']='fixed_recall_verified_candidates_with_moment_targets'
    (tr/'responsibility_summary.json').write_text(json.dumps(repair,indent=2))
    geometry=tr/'repaired_geometry_000008.npz'
    validation.append(dict(run=name,train_renders=12,heldout_renders=4,checkpoint_unchanged=True,heldout_replay_geometry=str(geometry),heldout_replay_geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),selection_exact=True))
(root/'main_method_receipt.json').write_text(json.dumps(method,indent=2))
(root/'main_validation.json').write_text(json.dumps(validation,indent=2))
files=[root/x for x in ['recall_comparison.json','heldout_recall_comparison.json','recall_queue_receipt.json','main_method_receipt.json','main_validation.json']]
for name in runs+[('heldout_'+n) for n in runs]:
    files.extend(p for p in (root/name).rglob('*') if p.is_file() and (p.suffix=='.json' or (p.suffix=='.png' and (p.name.endswith('_final.png') or p.name.endswith('_white.png') or p.name.endswith('_alpha.png')))))
manifest=[dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
(root/'main_evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(root/'main_evidence.tar.gz','w:gz') as tar:
    for p in files+[root/'main_evidence_manifest.json']:
        tar.add(p,arcname=str(p.relative_to(root)))
print(json.dumps(dict(files=len(files),bytes=(root/'main_evidence.tar.gz').stat().st_size)))
