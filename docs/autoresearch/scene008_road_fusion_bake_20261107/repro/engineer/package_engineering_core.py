"""Light CPU archive: persistent assets, full optimizer states, code and receipts.

Dense RGB/cache/targets stay preserved remotely and are separately inventoried.
Frozen scene checkpoints remain external dependencies pinned by each manifest.
"""
from pathlib import Path
import json
import shutil
import tarfile
from shared_fit_common import RUN,read,write,sha


def main():
    base=RUN/'engineer';ledger=read(base/'pipeline_compute_ledger_ABC.json')
    assert ledger['total_native_forwards']==2590 and ledger['total_native_backwards']==2528
    for name in ('direct','fusion','edgeblend'):
        directory=base/f'canonical_{name}_v1';receipt=read(directory/'parity/receipt.json')
        assert receipt['forwards']==10 and receipt['all_float_exact'] and receipt['all_PNG_byte_exact']
    references=base/'root_reference_receipts';references.mkdir(exist_ok=False)
    origins={}
    for folder,name in (('full49','AB_full49_receipt.json'),('edgeblend49','C_full49_receipt.json')):
        path=RUN/'results'/folder/'receipt.json';shutil.copy2(path,references/name);origins[name]=dict(path=str(path),sha256=sha(path))
    write(references/'origins.json',origins)
    included=set();excluded={}
    npz_names={'new_road_canonical_candidate.npz','material_override.npz','qualified_parents.npz','supervision_masks.npz'}
    roots=('material_fit_v1','smoke_engineering_placeholder','material_fit_edgeblend_v1',
        'canonical_direct_v1','canonical_fusion_v1','canonical_edgeblend_v1','root_reference_receipts')
    for path in base.iterdir():
        if path.is_file() and path.suffix in ('.py','.md','.json'):included.add(path)
    for name in roots:
        for path in (base/name).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:continue
            if path.suffix in ('.py','.md','.json','.pt') or path.name in npz_names:included.add(path)
            elif path.suffix in ('.png','.npz'):excluded[str(path.relative_to(base))]=dict(bytes=path.stat().st_size,sha256=sha(path),remote_preserved=True)
    excluded_path=base/'dense_artifacts_remote_inventory.json';write(excluded_path,dict(files=excluded,
        reason='lightweight delivery request; dense frame/cache/temporary fixture arrays remain unchanged on AutoDL',
        scientific_targets='target manifests hash all immutable NPZ; targets agent/root delivers those independently',
        frozen_V44='original checkpoints and dependencies remain pinned external scene-manifest inputs',AutoDL_shutdown=False))
    included.add(excluded_path)
    inventory={str(path.relative_to(base)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(included)}
    index=base/'engineering_core_manifest.json';write(index,dict(files=inventory,scene_status='GEOMETRY_SUPPORT_RESEARCH_CANDIDATE',
        adoption_claim=False,human_verdict=None,ledger_sha256=sha(base/'pipeline_compute_ledger_ABC.json')))
    archive=base/'engineering_core_delivery_v1.tar.gz'
    if archive.exists():raise FileExistsError('Keep previous delivery archive')
    with tarfile.open(archive,'w:gz') as tar:
        for path in sorted(included|{index}):tar.add(path,arcname=str(path.relative_to(base)),recursive=False)
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(inventory),
        inventory_sha256=sha(index),dense_files_preserved_remotely=len(excluded))))


if __name__=='__main__':main()
