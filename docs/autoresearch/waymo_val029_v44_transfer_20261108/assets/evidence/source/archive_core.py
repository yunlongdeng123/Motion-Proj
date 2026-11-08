"""Archive immutable reconstruction and data inputs before closing the GPU host."""
from pathlib import Path
import tarfile,hashlib,json,subprocess,time
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val029_cpu_20261108')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    paths=[]
    for name in ['engine','api','source','inputs/native','revealed_prior','revealed_surface','revealed_material','revealed_material_r2','semantic_all158','world_road','road_gate32_r2','road_gates']:
        p=ROOT/name
        if p.exists():paths.extend((q,str(q.relative_to(ROOT))) for q in p.rglob('*') if q.is_file() and not q.is_symlink() and '__pycache__' not in q.parts and q.suffix not in ['.pyc'])
    for p in (ROOT/'formal10k').iterdir():
        if p.is_file() and (p.suffix in ['.json','.jsonl'] or p.name.endswith('10000.pth')):paths.append((p,str(p.relative_to(ROOT))))
    for p in ROOT.glob('*.json'):paths.append((p,p.name))
    for name in ['inputs','masks_all_resolved_r2']:
        for p in (PREP/name).rglob('*'):
            if p.is_file() and not p.is_symlink():paths.append((p,'cpu_inputs/'+str(p.relative_to(PREP))))
    deps={'sh_asset.py':ROOT.parent/'scene008_a6_asset_controls_20261105/source/sh_asset.py','full_pvd.py':ROOT.parent/'scene008_a6_complete_asset_20261105/source/full_pvd.py','surface_lod_asset.py':ROOT.parent/'scene008_a6_insert_artifacts_20261105/source/scale/surface_lod_asset.py','fit_surface.py':ROOT.parent/'scene008_a6_insert_artifacts_20261105/surface_fit/source/fit_surface.py'}
    paths.extend((p,'inherited_source/'+name) for name,p in deps.items());paths=list({name:(p,name) for p,name in paths}.values())
    manifest=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for p,n in paths];(ROOT/'core_file_manifest.json').write_text(json.dumps(manifest,indent=2));archive=ROOT/'recovery_core.tar'
    with tarfile.open(archive,'w') as t:
        for p,n in paths:t.add(p,arcname=n,recursive=False)
        t.add(ROOT/'core_file_manifest.json',arcname='core_file_manifest.json')
    receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(manifest),purpose='10K complete state, current new-scene assets, CPU source bank and masks, native engine/API and inherited helper sources',raw_tfrecord='referenced by immutable SHA in CPU receipt; not duplicated in this archive',unix=time.time());(ROOT/'recovery_core_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
