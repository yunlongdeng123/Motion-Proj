"""Download pinned FreeInpaint object-removal dependencies; never imports a model.

Requires huggingface_hub with hf_xet (or curl fallback). Metadata files come from the HF models API
with blobs=true. Large files are checked against upstream LFS SHA256, small files
against Git blob IDs. Incomplete downloads remain resumable; existing files are
never silently overwritten. No experiment or inference is launched.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path(os.environ.get('V42_WEIGHTS_ROOT', '/root/autodl-tmp/models/worldsim_v82_v42_freeinpaint_20261024')).absolute()
REVISIONS = {
    'jyproris/FreeInpaint': 'd9bf8acc084bd3fdcb3e82f01b6af9fd73e5b157',
    'stable-diffusion-v1-5/stable-diffusion-inpainting': '8a4288a76071f7280aedbdb3253bdb9e9d5d84bb',
    'stable-diffusion-v1-5/stable-diffusion-v1-5': '451f4fe16113bff5a5d2269ed5ad43b0592e9a14',
    'latent-consistency/lcm-lora-sdv1-5': 'cf2fced511dbe7e26c8d1d397e728fbab875db4b',
    'h94/IP-Adapter': '018e402774aeeddd60609b4ecdb7e298259dc729',
}


def selected(repo, path):
    if repo == 'jyproris/FreeInpaint':
        return path.startswith('checkpoints/') or path == 'README.md'
    if repo.endswith('/stable-diffusion-inpainting'):
        return (path.endswith(('.json', '.txt', '.md')) or
                path in ['safety_checker/pytorch_model.bin', 'text_encoder/pytorch_model.bin',
                         'unet/diffusion_pytorch_model.bin', 'vae/diffusion_pytorch_model.bin'] or
                (path.endswith('.fp16.safetensors') and not path.startswith('safety_checker/')))
    if repo.endswith('/stable-diffusion-v1-5'):
        return path.startswith('tokenizer/') or path == 'README.md'
    if repo.startswith('latent-consistency/'):
        return path in ['pytorch_lora_weights.safetensors', 'README.md']
    if repo == 'h94/IP-Adapter':
        return path in ['models/ip-adapter_sd15.bin', 'models/image_encoder/config.json',
                        'models/image_encoder/model.safetensors', 'README.md']
    return False


def make_manifest():
    entries = []
    ROOT.mkdir(parents=True, exist_ok=True)
    for repo, revision in REVISIONS.items():
        metadata = ROOT / (repo.replace('/', '--') + '.metadata.json')
        if metadata.exists():
            data = json.loads(metadata.read_text(encoding='utf-8'))
        else:
            # Official pinned metadata remains the authority even when a mirror
            # supplies the model bytes. No model code is imported here.
            url = 'https://huggingface.co/api/models/' + repo + '/revision/' + revision + '?blobs=true'
            with urllib.request.urlopen(url, timeout=40) as response:
                data = json.load(response)
            assert data['sha'] == revision, (repo, data['sha'])
            metadata.write_text(json.dumps(data, indent=2), encoding='utf-8')
        assert data['sha'] == revision, (repo, data['sha'])
        for item in data['siblings']:
            if not selected(repo, item['rfilename']):
                continue
            algo, expected = ('sha256', item['lfs']['sha256']) if 'lfs' in item else ('git_blob_sha1', item['blobId'])
            entries.append(dict(repo=repo, revision=revision, file=item['rfilename'], size=item['size'], algorithm=algo, expected_hash=expected))
    return entries


def digest(path, entry):
    h = hashlib.sha256() if entry['algorithm'] == 'sha256' else hashlib.sha1()
    if entry['algorithm'] == 'git_blob_sha1':
        h.update(('blob ' + str(entry['size']) + '\0').encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(entry):
    cache = ROOT / 'hub' / ('models--' + entry['repo'].replace('/', '--'))
    path = cache / 'snapshots' / entry['revision'] / entry['file']
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    start = time.time()
    if path.exists():
        assert path.stat().st_size == entry['size'], str(path) + ': existing size mismatch'
        actual = digest(path, entry)
        assert actual == entry['expected_hash'], str(path) + ': existing hash mismatch'
    else:
        print(json.dumps(dict(event='download_start', file=entry['repo'] + '/' + entry['file'], bytes=entry['size'])), flush=True)
        url = 'https://huggingface.co/' + entry['repo'] + '/resolve/' + entry['revision'] + '/' + entry['file']
        if os.environ.get('V42_DOWNLOAD_BACKEND') == 'huggingface':
            from huggingface_hub import hf_hub_download
            downloaded = Path(hf_hub_download(repo_id=entry['repo'], filename=entry['file'], revision=entry['revision'], local_dir=cache / 'snapshots' / entry['revision']))
            assert downloaded == path
            assert path.stat().st_size == entry['size'], str(path) + ': download size mismatch'
            actual = digest(path, entry)
            assert actual == entry['expected_hash'], str(path) + ': download hash mismatch'
        else:
            subprocess.run(['curl', '-fLsS', '--retry', '6', '--retry-delay', '3', '--connect-timeout', '40', '--max-time', '7200', '--continue-at', '-', '--output', str(part), url], check=True)
            assert part.stat().st_size == entry['size'], str(part) + ': download size mismatch'
            actual = digest(part, entry)
            assert actual == entry['expected_hash'], str(part) + ': download hash mismatch'
            part.rename(path)
    result = dict(entry, path=str(path), actual_hash=actual, verified=True, seconds=round(time.time()-start, 2))
    print(json.dumps(dict(event='verified', file=entry['repo'] + '/' + entry['file'], bytes=entry['size'], seconds=result['seconds'])), flush=True)
    return result


def main():
    entries = make_manifest()
    total = sum(e['size'] for e in entries)
    (ROOT / 'download_plan.json').write_text(json.dumps(dict(revisions=REVISIONS, files=entries, total_bytes=total), indent=2))
    remaining = sum(e['size'] for e in entries if not (ROOT / 'hub' / ('models--' + e['repo'].replace('/', '--')) / 'snapshots' / e['revision'] / e['file']).exists())
    assert shutil.disk_usage(ROOT).free > remaining + 10 * 1024**3, 'Need 10 GiB spare after planned downloads'
    print(json.dumps(dict(event='plan', files=len(entries), total_bytes=total, pid=os.getpid())), flush=True)
    receipt = dict(status='running', started_at=time.time(), pid=os.getpid(), inference_run=False, files=[])
    (ROOT / 'download_receipt.json').write_text(json.dumps(receipt, indent=2))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(fetch, entries):
                receipt['files'].append(result)
                (ROOT / 'download_receipt.json').write_text(json.dumps(receipt, indent=2))
        for repo, revision in REVISIONS.items():
            refs = ROOT / 'hub' / ('models--' + repo.replace('/', '--')) / 'refs'
            refs.mkdir(parents=True, exist_ok=True)
            (refs / 'main').write_text(revision)
        alias = ROOT / 'hub/models--runwayml--stable-diffusion-inpainting'
        target = ROOT / 'hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting'
        if not alias.exists():
            alias.symlink_to(target, target_is_directory=True)
        checkpoints = ROOT / 'code/checkpoints'
        if not checkpoints.exists():
            checkpoints.symlink_to(ROOT / 'hub/models--jyproris--FreeInpaint/snapshots' / REVISIONS['jyproris/FreeInpaint'] / 'checkpoints', target_is_directory=True)
        receipt.update(status='complete', total_bytes=total, file_count=len(entries), completed_at=time.time())
    except Exception as e:
        receipt.update(status='failed', error=repr(e))
        raise
    finally:
        (ROOT / 'download_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
