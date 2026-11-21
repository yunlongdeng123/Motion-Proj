"""Stdlib-only archive inventory. No evaluation, rendering, or result mutation."""
import hashlib
import json
from pathlib import Path
import tarfile

D = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/diagnostic/color_optimality')
P = D.parents[1]
packet = D / 'numeric_color_packet.tar.gz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


with tarfile.open(str(packet), 'r:gz') as tar:
    members = []
    for member in tar.getmembers():
        if member.isfile():
            members.append({'path': member.name, 'bytes': member.size,
                            'SHA256': hashlib.sha256(tar.extractfile(member).read()).hexdigest()})
    receipt = json.load(tar.extractfile('numeric_receipt.json'))
    identity = json.load(tar.extractfile('objective_identity.json'))
    export = json.load(tar.extractfile('export_manifest.json'))
    source_modules = json.load(tar.extractfile('code/source_manifest.json'))

known = dict(receipt['source_SHA256'])
known.update(identity['source_SHA256'])
for item in source_modules.values():
    known[item['original_path']] = item['SHA256']
external = set(known)
# Complete persistent parent asset is required; the CI and optimizer files are patches.
external.update(str(f) for f in (P / 'appearance_fit/asset').rglob('*') if f.is_file())
teacher = P.parent / 'road_common1152_20261121/teacher'
for raw in (86, 110):
    external.add(str(teacher / ('raw%03d' % raw) / 'supervision.npz'))

deps = []
for name in sorted(external):
    f = Path(name)
    deps.append({'path': name, 'bytes': f.stat().st_size,
                 'SHA256': known.get(name) or sha(f),
                 'SHA256_origin': 'frozen_source_receipt' if name in known else 'archive_inventory_read_only',
                 'included_in_packet': False})

evidence = []
for row in export['rows']:
    f = Path(row['packet'])
    evidence.append({'raw': row['raw'], 'path': str(f), 'bytes': f.stat().st_size,
                     'SHA256': sha(f), 'included_in_packet': False,
                     'role': 'remote-only float render evidence; PNGs included'})

out = {
    'schema': 'numeric_color_restore_dependencies.v1',
    'id': 'numeric_color', 'complete': True,
    'packet': {'path': str(packet), 'bytes': packet.stat().st_size, 'SHA256': sha(packet)},
    'included_files': members,
    'external_dependencies': deps,
    'remote_only_evidence': evidence,
    'restore': {
        'parent_asset': str(P / 'appearance_fit/asset'),
        'CI_patch': 'accepted_CI.npz',
        'optimizer_state': 'solver/final_workspace.npz',
        'RNG_state': 'solver/final_RNG.pt',
        'normalizers_and_mask_target_signatures': 'objective_identity.json',
        'loader_source': 'code/probe_color_optimality.py',
        'environment_ABI': receipt.get('ABI'),
        'requires_original_renderer_runtime': True,
        'scope': 'Exact declared source files plus complete persistent formal parent asset. '
                 'The small packet is a patch/state archive, not a standalone renderer distribution.'
    },
    'formal_numeric': {'FG': receipt['FG'], 'accepted': receipt['accepted']},
    'formal_R2_unchanged': True, 'additional_GPU_calls': 0,
    'existing_packet_modified': False,
}
target = D / 'restore_dependencies.json'
target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'path': str(target), 'files': len(members), 'external_dependencies': len(deps),
                  'bytes': target.stat().st_size, 'SHA256': sha(target)}))
