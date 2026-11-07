"""Explicit lean per-asset backup inventory; dense parity RGB stays remote."""
from shared_fit_common import RUN,write,sha


def main():
    base=RUN/'engineer';result={}
    for name in ('direct','fusion','edgeblend'):
        folder=base/f'canonical_{name}_v1';keep={};omit={}
        for path in folder.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:continue
            entry=dict(bytes=path.stat().st_size,sha256=sha(path))
            if path.suffix in ('.json','.py','.md') or path.name in ('new_road_canonical_candidate.npz','material_override.npz'):
                keep[str(path.relative_to(base))]=entry
            else:omit[str(path.relative_to(base))]=entry
        result[name]=dict(include=keep,remote_dense=omit,lean_bytes=sum(x['bytes'] for x in keep.values()),
            scope='geometry/material NPZ, manifests, runtime/source, audit/parity receipts; dense float/PNG has separate report copy')
    target=base/'lean_backup_list.json';write(target,dict(assets=result,adoption_claim=False,human_verdict=None,remote_dense_preserved=True))
    print(dict(path=str(target),sha256=sha(target),asset_lean_bytes={name:info['lean_bytes'] for name,info in result.items()}))


if __name__=='__main__':main()
