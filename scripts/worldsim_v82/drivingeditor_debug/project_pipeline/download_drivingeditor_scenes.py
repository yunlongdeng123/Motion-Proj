"""Download only four author-named Waymo scenes into the existing raw layout."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import urllib.request

ROOT = Path('/root/autodl-tmp/datasets/waymo_v1_4_3')
PROJECT = Path('/root/autodl-tmp/drivingeditor_repro')
MANIFEST = PROJECT / 'data_manifest.json'
REPO = 'AnnaZhang/waymo_open_dataset_v_1_4_3'
REV = '8a6fccd2b022077308c167a4d3c6f646d6e5e65b'
AUTHOR = 'https://github.com/fudan-zvg/PVG/issues/26#issuecomment-2182469515'
SELECTED = [(145, '0145050'), (147, '0147030'), (158, '0158150'), (17, '0017085')]
LOCK = threading.RLock()
STATE = {}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHUNK = 64 * 1024**2

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save():
    with LOCK:
        STATE['updated_utc'] = utc()
        temp = MANIFEST.with_suffix('.json.tmp')
        temp.write_text(json.dumps(STATE, indent=2) + '\n')
        os.replace(temp, MANIFEST)

def event(kind, **kw):
    with LOCK:
        record = dict(utc=utc(), event=kind, **kw)
        print(json.dumps(record), flush=True)
        with (PROJECT/'logs/data_download_events.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(8*1024**2)
            if not block: break
            digest.update(block)
    return digest.hexdigest()

def transfer(row):
    destination = ROOT / row['relative_path']
    assert destination.parent == ROOT/'individual_files/training'
    partial = destination.with_suffix(destination.suffix+'.drivingeditor.part')
    if destination.is_symlink() or partial.is_symlink():
        raise RuntimeError('Refuse symlink destination')
    if destination.exists():
        actual = sha256(destination)
        if destination.stat().st_size != row['size_bytes'] or actual != row['sha256']:
            raise RuntimeError('Existing destination does not match; preserved unchanged')
        row.update(status='verified', verified_sha256=actual, verified_utc=utc())
        save()
        return
    if partial.exists() and not row.get('owned_partial'):
        raise RuntimeError('Existing partial not owned by this run; preserved unchanged')
    if not partial.exists():
        with partial.open('xb'): pass
        row['owned_partial'] = str(partial)
        save()
    while partial.stat().st_size < row['size_bytes']:
        start = partial.stat().st_size
        end = min(start+CHUNK, row['size_bytes'])-1
        for attempt in range(1, 7):
            if shutil.disk_usage(ROOT).free < 25*1024**3 + (end-start+1):
                raise RuntimeError('Less than 25 GiB reserve; partial retained')
            request = urllib.request.Request(row['mirror_url'], headers={
                'Range': f'bytes={start}-{end}',
                'Accept-Encoding': 'identity',
                'User-Agent': 'DrivingEditor-data-preparation/1.0',
            })
            try:
                with OPENER.open(request, timeout=90) as response:
                    content_range = response.headers.get('Content-Range', '')
                    expected = f'bytes {start}-{end}/{row["size_bytes"]}'
                    if response.status != 206 or content_range != expected:
                        raise RuntimeError(f'Range mismatch: {response.status} {content_range}')
                    payload = response.read(end-start+1)
                    if len(payload) != end-start+1:
                        raise RuntimeError('Incomplete range; no partial modification')
                with partial.open('ab') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                row.update(status='downloading', bytes_present=partial.stat().st_size)
                save()
                event('range_complete', scene_id=row['scene_id'], bytes_present=row['bytes_present'], size_bytes=row['size_bytes'])
                break
            except Exception as exc:
                event('range_retry', scene_id=row['scene_id'], attempt=attempt, error=str(exc))
                if attempt == 6: raise
                time.sleep(min(attempt*2, 12))
    row['status'] = 'verifying'
    save()
    actual = sha256(partial)
    if partial.stat().st_size != row['size_bytes'] or actual != row['sha256']:
        raise RuntimeError('Downloaded size or SHA256 mismatch; partial preserved')
    os.link(partial, destination)  # Atomic no-clobber publication on data filesystem.
    partial.unlink()
    row.update(status='verified', bytes_present=row['size_bytes'], verified_sha256=actual, verified_utc=utc(), raw_path=str(destination))
    save()
    event('file_verified', scene_id=row['scene_id'], raw_path=str(destination), sha256=actual)

def main():
    global STATE
    tree=json.loads(Path('/root/autodl-tmp/dggt/.preparation/hf_waymo_training_tree.json').read_text())
    files=sorted((r for r in tree if r['path'].endswith('.tfrecord')), key=lambda r:r['path'])
    assert len(files)==798
    if MANIFEST.exists():
        STATE=json.loads(MANIFEST.read_text())
        if STATE['repository']!=REPO or STATE['revision']!=REV or [(r['global_training_index'],r['scene_id']) for r in STATE['files']]!=SELECTED:
            raise RuntimeError('Existing manifest identity mismatch')
    else:
        rows=[]
        for idx, scene in SELECTED:
            r=files[idx]
            assert re.fullmatch('[0-9a-f]{64}',r['lfs']['oid'])
            rows.append(dict(scene_id=scene,global_training_index=idx,relative_path=r['path'],
                size_bytes=r['size'],sha256=r['lfs']['oid'],status='pending',bytes_present=0,
                mirror_url=f'https://hf-mirror.com/datasets/{REPO}/resolve/{REV}/{r["path"]}',
                upstream_url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{r["path"]}'))
        STATE=dict(status='downloading',created_utc=utc(),repository=REPO,revision=REV,
            source_kind='Third-party mirror of Waymo v1.4.3, not official publisher',
            author_segment_evidence=AUTHOR,
            mapping_evidence='Author response names the four dynamic segments; official converter concatenates prefix0 + lexicographically sorted full-training file_idx padded3 + frame_idx padded3. Full 798-entry local list matches pinned mirror tree.',
            total_files=4,total_bytes=sum(r['size_bytes'] for r in rows),files=rows)
        save()
    event('started',total_files=4,total_bytes=STATE['total_bytes'])
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(transfer,row) for row in STATE['files']]
            for future in concurrent.futures.as_completed(futures): future.result()
    except Exception as exc:
        STATE.update(status='failed',error=str(exc)); save(); raise
    STATE.update(status='completed',finished_utc=utc());save()
    event('completed',total_bytes=STATE['total_bytes'])

if __name__=='__main__': main()
