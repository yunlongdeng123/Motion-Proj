"""Resume the official SAM checkpoint with verified HTTP byte ranges."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import requests

root = Path('/root/autodl-tmp/drivingeditor_repro')
path = root / 'weights/sam_vit_h_4b8939.pth'
url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
size = 2564550879
etag = '"11bf577cf1da9493a48d7a571ba83cd5-306"'
partial = path.with_name(path.name + '.partial')
prefix = partial.stat().st_size if partial.exists() else 0
chunks = []
chunk_size = 64 * 1024 * 1024
for start in range(prefix, size, chunk_size):
    end = min(size, start + chunk_size) - 1
    chunks.append((start, end, path.with_name(path.name + f'.range-{start}-{end}')))

def get_chunk(chunk):
    start, end, output = chunk
    for attempt in range(5):
        have = output.stat().st_size if output.exists() else 0
        assert have <= end - start + 1
        if have == end - start + 1:
            return chunk
        session = requests.Session()
        session.trust_env = False
        try:
            with session.get(url, headers={'Range': f'bytes={start+have}-{end}', 'If-Match': etag},
                             stream=True, timeout=(15, 40)) as response:
                assert response.status_code == 206, response.status_code
                assert response.headers['Content-Range'] == f'bytes {start+have}-{end}/{size}'
                assert response.headers['ETag'] == etag
                with output.open('ab') as stream:
                    for data in response.iter_content(1024 * 1024):
                        stream.write(data)
            assert output.stat().st_size == end - start + 1
            print(json.dumps({'completed_range': [start, end]}), flush=True)
            return chunk
        except Exception as exc:
            print(json.dumps({'range_start': start, 'attempt': attempt, 'error': str(exc)}), flush=True)
            if attempt == 4:
                raise
            time.sleep(2)
        finally:
            session.close()

if not path.exists():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(get_chunk, chunks))
    assert (partial.stat().st_size if partial.exists() else 0) == prefix
    with partial.open('ab') as output:
        for _, _, chunk_file in chunks:
            with chunk_file.open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    assert partial.stat().st_size == size
    digest = hashlib.sha256()
    with partial.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    sha256 = digest.hexdigest()
    # Full hash independently corroborated by HF's original SAM checkpoint mirror.
    expected_sha256 = 'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e'
    assert sha256 == expected_sha256, sha256
    partial.replace(path)
    # Delete only this downloader's verified chunks after successful publication.
    for chunk_file in path.parent.glob(path.name + '.range-*'):
        assert chunk_file.resolve().parent == path.parent.resolve()
        chunk_file.unlink()
    (root / 'logs/sam_weight_provenance.json').write_text(json.dumps({
        'url': url, 'size': size, 'etag': etag, 'sha256': sha256,
        'resumed_prefix_bytes': prefix, 'http_ranges_verified': True,
        'checksum_reference': 'https://huggingface.co/ybelkada/segment-anything/blob/487029d784104fb1a9523f7045226a27f8aa1c0d/checkpoints/sam_vit_h_4b8939.pth',
    }, indent=2) + '\n')
    print('SAM_VERIFIED ' + sha256, flush=True)
