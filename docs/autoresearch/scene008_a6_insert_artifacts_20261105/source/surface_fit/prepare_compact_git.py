"""Curate small reviewed source/evidence for Git, omitting matrices and replays."""
import hashlib,json,shutil
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    dest=HERE/'compact_git_v1';dest.mkdir(exist_ok=False)
    pairs=[]
    def add(source,target,role):pairs.append((ROOT/source,Path(target),role))
    for name in ['fit_surface.py','combine_rgba.py','fit_observed.py','evaluate_combined.py','audit_posterior.py','render_surface.py','render_final.py','prepare_review.py','prepare_compact_git.py']:
        add('surface_fit/'+name,'source/surface_fit/'+name,'source')
    for name in ['diagnose_shadow_layers.py','conditional_geometry.py']:
        add('shadow/'+name,'source/shadow/'+name,'source')
    for name in ['source_audit.json','visual_ROI_attribution.json','qa_receipt_v2.json','SOURCE_AUDIT.md']:
        add('shadow/'+name,'evidence/shadow/'+name,'audit')
    for source,name in [('results/conditional_geometry.json','conditional_geometry_v2.json'),('results/receipt.json','layer_attribution_receipt.json')]:
        add('shadow/'+source,'evidence/shadow/'+name,'native_forward_evidence')
    for name in ['architecture.png','shadow_source_controls.png']:
        add('shadow/'+name,'figures/shadow/'+name,'reviewed_figure')
    for source,name in [
        ('results/receipt.json','depth_receipt.json'),('results/tail_audit.json','tail_audit.json'),
        ('combined_v1/receipt.json','combined_receipt.json'),('combined_v1/review.json','review.json'),
        ('combined_v1/REVIEW.md','REVIEW.md'),('combined_v1/evaluation_roles.json','evaluation_roles.json'),
        ('combined_v1/compute_ledger.json','compute_ledger.json'),
        ('combined_v1/observed_fit/collection.json','collection.json'),
        ('combined_v1/observed_fit/smoke/receipt.json','smoke_receipt.json'),
        ('combined_v1/observed_fit/formal/receipt.json','formal_receipt.json'),
        ('combined_v1/observed_fit/evaluation/receipt.json','evaluation_receipt.json'),
        ('combined_v1/observed_fit/evaluation/summary.json','summary.json'),
        ('combined_v1/observed_fit/evaluation/HR_summary.json','HR_summary.json')]:
        add('surface_fit/'+source,'evidence/surface/'+name,'fixed_experiment_evidence')
    for name in ['architecture.png','depth_posterior.png']:
        add('surface_fit/combined_v1/'+name,'figures/surface/'+name,'reviewed_figure')
    for raw in [106,113,130]:
        add(f'surface_fit/combined_v1/observed_fit/evaluation/raw{raw}/black_HR_comparison.png',f'figures/surface/raw{raw}_black_HR_comparison.png','actual_native_render')
    for raw,name in [(113,'black_native480_comparison'),(113,'delete_crop480_comparison'),(130,'delete_crop480_comparison')]:
        add(f'surface_fit/combined_v1/observed_fit/evaluation/raw{raw}/{name}.png',f'figures/surface/raw{raw}_{name}.png','actual_native_render')
    files=[]
    for source,target,role in pairs:
        if not source.is_file():raise FileNotFoundError(source)
        out=dest/target;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,out)
        assert sha(source)==sha(out)
        files.append(dict(path=target.as_posix(),bytes=out.stat().st_size,sha256=sha(out),role=role,original_workspace_path=source.relative_to(ROOT).as_posix()))
    asset=HERE/'combined_v1/observed_fit/formal/complete_a6.npz'
    manifest=dict(status='PASS_COMPACT_SOURCE_EVIDENCE_BYTE_VERIFIED',files=files,total_bytes=sum(x['bytes'] for x in files),
        excluded=['*.tgz','design.npz','native480_replay.npz','HR_black_replay.npz','checkpoints/full_state.pt','posterior.npz'],
        canonical_asset_optional=dict(path=asset.relative_to(ROOT).as_posix(),bytes=asset.stat().st_size,sha256=sha(asset)),
        original_numeric_receipts_preserved=True,interpretation_correction='evidence/surface/evaluation_roles.json',
        standalone_runtime_included=False,requires='Original frozen controls/runtime, original observation bank and original campaign sources',
        no_adoption_or_untouched_generalization_claim=True)
    (dest/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (dest/'README.md').write_text('''Small source/evidence selection for this fixed experiment. All copies are SHA256 verified in MANIFEST.json.

The original numerical receipts are retained, including their historical final-only evaluation flag. Read evidence/surface/evaluation_roles.json: HELD9 was excluded from the recorded colour objective but previously used in representation diagnosis and internal development. These are internal comparisons, not an untouched generalization test.

Source scripts retain their exact executed campaign paths. Replaying them requires the existing frozen Runtime, original observations and source campaign assets; this folder is not a standalone scene export. The canonical asset is listed as an optional 2.8MB file in MANIFEST.json. Large matrices, state checkpoints, full float32 replays and the in-flight archive are intentionally outside this Git selection.

Surface REVIEW.md relative image links refer to the original working layout; reviewed figures are copied into figures/surface here. Geometry changed for the depth posterior; only the later colour fit freezes it. Frozen background parameters do not imply unchanged composited images. No representation adoption is asserted.
''',encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'],files=len(files),bytes=manifest['total_bytes'],path=str(dest))))

if __name__=='__main__':main()
