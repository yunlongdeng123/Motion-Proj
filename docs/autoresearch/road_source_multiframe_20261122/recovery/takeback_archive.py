"""取回已完成的 curated archive，并在指定 outputs 目录逐文件验证。"""
from pathlib import Path, PurePosixPath
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import tarfile
from restore_manifest import sha, verify_archive

HERE = Path(__file__).resolve().parent
REMOTE_ROOT = '/root/autodl-tmp/drivingeditor_repro/campaign/road_source_multiframe_20261122_r3'
DEFAULT_DEST = HERE.parents[2] / 'outputs/road_source_multiframe_20261122_r3'


def download_parts(receipt, archive):
    remote_archive = receipt['archive']
    remote_folder = REMOTE_ROOT + '/takeback_parts/' + PurePosixPath(remote_archive).stem
    code = """from pathlib import Path
import hashlib,json
source=Path(%r); folder=Path(%r); folder.mkdir(parents=True,exist_ok=True)
rows=[]
with source.open('rb') as stream:
 for i,block in enumerate(iter(lambda:stream.read(8*1024*1024),b'')):
  p=folder/('part_%%03d'%%i); expected=hashlib.sha256(block).hexdigest()
  if not p.exists() or p.stat().st_size!=len(block) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
   temp=p.with_suffix('.tmp');temp.write_bytes(block);temp.replace(p)
  rows.append(dict(name=p.name,path=str(p),bytes=len(block),sha256=expected))
print(json.dumps(dict(parts=rows)))
""" % (remote_archive, remote_folder)
    encoded = base64.b64encode(code.encode()).decode()
    command = "/root/autodl-tmp/envs/motionproj/bin/python -c \"import base64;exec(base64.b64decode('" + encoded + "'))\""
    manifest = json.loads(subprocess.run(['ssh', 'wm-vgpu-1011', command], check=True, capture_output=True).stdout)
    folder = HERE / 'download_parts' / archive.name
    folder.mkdir(parents=True, exist_ok=True)
    def retrieve(row):
        if Path(row['name']).name != row['name']:
            raise ValueError('Invalid part filename')
        p = folder / row['name']
        if not p.exists() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            subprocess.run(['scp', '-q', 'wm-vgpu-1011:' + row['path'], str(p)], check=True, timeout=90)
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Downloaded chunk differs: ' + row['name'])
        return p
    with ThreadPoolExecutor(max_workers=2) as pool:
        parts = list(pool.map(retrieve, manifest['parts']))
    part = archive.with_suffix(archive.suffix + '.part')
    with part.open('wb') as out:
        for p in parts:
            with p.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''):
                    out.write(block)
    if part.stat().st_size != receipt['archive_bytes'] or sha(part) != receipt['archive_sha256']:
        raise ValueError('Assembled archive differs from remote verified archive')
    part.replace(archive)
    (folder / 'parts.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


def write_new_or_same(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Preserve differing existing delivery file: ' + str(path))
        return
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_bytes(payload)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default=str(DEFAULT_DEST))
    parser.add_argument('--chunked', action='store_true', help='Use verified 8MiB chunks for this host large-file SCP issue')
    args = parser.parse_args()
    dest = Path(args.destination).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    raw = subprocess.run(['ssh', 'wm-vgpu-1011', 'cat ' + REMOTE_ROOT + '/recovery/archive_receipt.json'],
                         check=True, capture_output=True).stdout
    receipt = json.loads(raw)
    if not receipt.get('complete') or not receipt.get('verified_every_member'):
        raise ValueError('Remote archive has no completed verification receipt')
    remote_archive = PurePosixPath(receipt['archive'])
    remote_archive.relative_to(PurePosixPath(REMOTE_ROOT))
    archive = dest / remote_archive.name
    if not archive.exists():
        if args.chunked:
            download_parts(receipt, archive)
        else:
            part = archive.with_suffix(archive.suffix + '.part')
            subprocess.run(['scp', '-q', 'wm-vgpu-1011:' + str(remote_archive), str(part)], check=True, timeout=90)
            if part.stat().st_size != receipt['archive_bytes'] or sha(part) != receipt['archive_sha256']:
                raise ValueError('Downloaded archive differs from remote verified file')
            part.replace(archive)
    if archive.stat().st_size != receipt['archive_bytes'] or sha(archive) != receipt['archive_sha256']:
        raise ValueError('Existing delivery archive differs from remote verified file')
    manifest = verify_archive(archive)
    exports = {
        'restore_manifest.json': 'restore_manifest.json',
        'run/recovery/resource_ledger_cumulative.json': 'resource_ledger_cumulative.json',
        'run/recovery/complete_result_matrix.json': 'complete_result_matrix.json',
        'run/recovery/packet_state_audit.json': 'packet_state_audit.json',
        'run/recovery/restore_manifest.py': 'restore_manifest.py',
    }
    with tarfile.open(archive, 'r:gz') as tar:
        for member, filename in exports.items():
            write_new_or_same(dest / filename, tar.extractfile(member).read())
    write_new_or_same(dest / 'archive_receipt.json', raw)
    compact_raw = subprocess.run(['ssh', 'wm-vgpu-1011', 'cat ' + REMOTE_ROOT + '/restore_manifest_compact.json'],
                                check=True, capture_output=True).stdout
    compact = json.loads(compact_raw)
    if compact['archive']['sha256'] != receipt['archive_sha256']:
        raise ValueError('Compact restore entry does not identify the verified archive')
    write_new_or_same(dest / 'restore_manifest_compact.json', compact_raw)
    result = dict(complete=True, remote_archive=str(remote_archive), local_archive=str(archive),
        archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
        every_archive_member_verified=True, payload_files=len(manifest['files']),
        exported_metadata=list(exports.values()) + ['restore_manifest_compact.json'],
        main_archive_role=compact['main_archive_role'],
        full_report_companion=compact['full_report_companion'],
        report_companion_role=compact['report_companion_role'],
        report_viewing_entry='Original HTTP/original HTML only; backup archives are restoration artifacts.',
        GPU_calls=0, power_operations=0)
    write_new_or_same(dest / 'takeback_receipt.json', (json.dumps(result, indent=2) + '\n').encode())
    write_new_or_same(dest / 'local_delivery.json', (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
