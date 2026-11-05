"""Collect compact review/replay artifacts; original remote run is retained."""
from pathlib import Path
import json,hashlib,shutil,tarfile
RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
OLD=RUN.parent/'scene008_a6_asset_controls_20261105'
DEST=RUN/'publication_inputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def copy(p,relative=None):
    dst=DEST/(relative if relative is not None else p.relative_to(RUN));dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
def main():
    DEST.mkdir(exist_ok=True)
    # Small contracts, model manifests, color solver states and standalone assets.
    for root in ('source','highres','highres_evaluation','generated_fit','output_adapter','candidates_raw106','backend_checks','hosted_generation','reviews','final_gallery','final_gallery_surface','final_gallery_formal_surface'):
        folder=RUN/root
        if not folder.exists():continue
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=p.relative_to(RUN);suffix=p.suffix.lower();name=p.name
            if suffix in ('.py','.json','.svg','.md','.mmd','.txt','.log','.csv') or '.bak' in name:
                copy(p)
            elif root.startswith('final_gallery') and suffix in ('.jpg','.png'):copy(p)
            elif root=='hosted_generation' and suffix=='.png':copy(p)
            elif root=='candidates_raw106' and (name in ('candidate_board.png','composed.png')):copy(p)
            elif suffix=='.npz' and name in ('complete_a6.npz','sphere_sigma_asset.npz','projective_surface_asset.npz','replay.npz','full_state.npz','asset.npz','surface_asset.npz'):
                copy(p)
            elif suffix in ('.png','.jpg') and (name.endswith('_objects.png') or name.endswith('_objects.jpg') or name.endswith('_black_crop.png') or name.startswith('comparison_')):copy(p)
    for p in RUN.glob('*.log'):copy(p)
    copy(OLD/'m2/complete_a6.npz','baseline/M2_complete_a6.npz')
    for p in (RUN/'inputs').glob('*.json'):copy(p)
    for p in (RUN/'inputs_hr').glob('*.json'):copy(p)
    # Keep a compact actual BUILD106 proposal interface; no HELD generation inputs.
    for root in ('inputs/raw106','inputs_hr/raw106'):
        for p in (RUN/root).glob('*'):
            if p.is_file() and p.name not in ('design_A_normalized.npz','design_W_premultiplied.npz','raster_original.npz','raster_hr.npz','real_rgb_original.png','real_rgb_hr.png'):
                copy(p)
    rows=[dict(path=str(p.relative_to(DEST)).replace('\\','/'),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DEST.rglob('*')) if p.is_file()]
    manifest=dict(status='COMPLETE_COMPACT_REVIEW_PACKAGE',files=rows,files_count=len(rows),original_run=str(RUN),
        large_fixed_scene_checkpoints_and_ray_designs_retained_remote=True,model_weights_not_redistributed=True)
    (DEST/'collection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    archive=RUN/'publication_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(DEST)),recursive=False)
    (RUN/'publication_inputs.tar.gz.sha256').write_text(sha(archive)+'  publication_inputs.tar.gz\n')
    print(json.dumps(dict(files=len(rows),bytes=archive.stat().st_size,sha256=sha(archive))),flush=True)
if __name__=='__main__':main()
