"""下载 DiffuEraser 推理必需文件；不导入模型、不启动推理。"""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import time
import subprocess
import requests

ROOT = Path('/root/autodl-tmp/models/worldsim_v77_diffueraser')
RUN = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-HYBRID-BG-20260927/r1')
FILES = {
    'lixiaowen/diffuEraser': ('diffuEraser', [
        'brushnet/config.json', 'brushnet/diffusion_pytorch_model.safetensors',
        'unet_main/config.json', 'unet_main/diffusion_pytorch_model.safetensors']),
    'stable-diffusion-v1-5/stable-diffusion-v1-5': ('stable-diffusion-v1-5', [
        'model_index.json', 'feature_extractor/preprocessor_config.json',
        'safety_checker/config.json', 'safety_checker/model.safetensors',
        'scheduler/scheduler_config.json', 'text_encoder/config.json',
        'text_encoder/model.safetensors', 'tokenizer/merges.txt',
        'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json',
        'tokenizer/vocab.json']),
    'wangfuyun/PCM_Weights': ('PCM_Weights', [
        'sd15/pcm_sd15_smallcfg_2step_converted.safetensors']),
    'stabilityai/sd-vae-ft-mse': ('sd-vae-ft-mse', [
        'config.json', 'diffusion_pytorch_model.safetensors']),
}

def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)

def download_sequential(entry):
    path = Path(entry['local_path'])
    part = path.with_suffix(path.suffix + '.part')
    path.parent.mkdir(parents=True, exist_ok=True)
    size = entry['expected_bytes']
    if path.exists():
        if path.stat().st_size != size:
            raise RuntimeError(f'Existing final file size mismatch: {path}')
        return str(path)
    for attempt in range(4):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > size:
                raise RuntimeError(f'Partial file exceeds expected size: {part}')
            if offset < size:
                headers = {'Range': f'bytes={offset}-'} if offset else {}
                # 避免过期的 CDN 重定向；来源始终固定到同一个官方 revision。
                url = entry['url'] + f'?download=true&attempt={int(time.time())}'
                with requests.get(url, stream=True, headers=headers, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                        raise RuntimeError('Server did not honor resume range; preserving partial file')
                    with part.open('ab' if offset else 'wb') as output:
                        for block in response.iter_content(4 * 1024 * 1024):
                            output.write(block)
            if part.stat().st_size != size:
                raise RuntimeError(f'Incomplete file: {part.stat().st_size}/{size}')
            part.replace(path)
            print(f'COMPLETE {entry["relative_path"]} {size}', flush=True)
            return str(path)
        except Exception as exc:
            print(f'RETRY {attempt+1} {entry["relative_path"]}: {type(exc).__name__}: {str(exc)[:160]}', flush=True)
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))

def download(entry):
    path = Path(entry['local_path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    size = entry['expected_bytes']
    if path.exists():
        assert path.stat().st_size == size
        return str(path)
    if size < 2_000_000:
        return download_sequential(entry)
    pending = path.with_suffix(path.suffix + '.aria2part')
    url = entry['url'] + f'?download=true&attempt={int(time.time())}'
    command = ['aria2c', '--console-log-level=warn', '--summary-interval=30',
               '--auto-file-renaming=false', '--allow-overwrite=false', '--continue=true',
               '--file-allocation=none', '--check-certificate=true', '--max-tries=4',
               '--retry-wait=3', '--timeout=90', '--connect-timeout=30', '-x', '8', '-s', '8',
               '--dir', str(path.parent), '--out', pending.name]
    env = dict(os.environ)
    if 'modelscope.cn' in url:
        for key in ['http_proxy','https_proxy','all_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']:
            env.pop(key, None)
    else:
        command.append('--all-proxy='+os.environ['HTTPS_PROXY'])
        # 官方仓库元数据经代理，官方 CDN 直连；避免大文件绕行本地隧道。
        command.append('--no-proxy=us.aws.cdn.hf.co,aws.cdn.hf.co,us.gcp.cdn.hf.co,cas-bridge.xethub.hf.co,cas-server.xethub.hf.co,cdn-lfs.huggingface.co,cdn-lfs-us-1.hf.co,cdn-lfs.hf.co')
    log = RUN/(entry['relative_path'].replace('/','_')+'.aria2.log')
    with log.open('a') as output:
        subprocess.run(command+[url], check=True, env=env, stdout=output, stderr=subprocess.STDOUT)
    assert pending.stat().st_size == size
    pending.replace(path)
    print(f'COMPLETE {entry["relative_path"]} {size}', flush=True)
    return str(path)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN / 'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        entries = []
        for repo, (folder, filenames) in FILES.items():
            response = requests.get(f'https://huggingface.co/api/models/{repo}', timeout=30)
            response.raise_for_status()
            revision = response.json()['sha']
            response = requests.get(f'https://huggingface.co/api/models/{repo}/tree/{revision}', params={'recursive':'true'}, timeout=30)
            response.raise_for_status()
            sizes = {item['path']: item['size'] for item in response.json() if item['type'] == 'file'}
            for filename in filenames:
                entries.append(dict(repo=repo, revision=revision, relative_path=f'{folder}/{filename}',
                    expected_bytes=sizes[filename], local_path=str(ROOT/folder/filename),
                    url=f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'))
        reused = []
        for filename in ['ProPainter.pth', 'raft-things.pth', 'recurrent_flow_completion.pth']:
            source = Path('/root/autodl-tmp/models/worldsim_v77_poc_propainter')/filename
            assert source.stat().st_size > 1_000_000
            target = ROOT/'propainter'/filename
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.symlink_to(source)
            reused.append(dict(path=str(target), source=str(source), bytes=source.stat().st_size))
        manifest = dict(task='WS-V77-HYBRID-BG-20260927', run='r1', files=entries, reused=reused,
            total_download_bytes=sum(e['expected_bytes'] for e in entries),
            excluded=['SD1.5 UNet/vae duplicate weights', 'training motion adapter', 'other PCM variants', 'TABE', 'CoTracker: first pilot uses LK'],
            inference_authorized_now=False, human_verdict=None)
        save(manifest_path, manifest)
    # 官方 README 提供同作者 ModelScope 源；保留 HF 来源记录。
    for entry in manifest['files']:
        if entry['repo'] == 'lixiaowen/diffuEraser' and entry['expected_bytes'] > 2_000_000:
            entry.setdefault('huggingface_url', entry['url'])
            filename = entry['relative_path'].split('/', 1)[1]
            entry['url'] = 'https://modelscope.cn/models/xingzi/diffuEraser/resolve/cc8ebdfb2a1eafbfe8196c73b6cbff696cfc9d6f/'+filename
            entry['download_source'] = 'official ModelScope alternative; expected byte size matches official metadata'
    save(manifest_path, manifest)
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, entry) for entry in manifest['files']]
        while True:
            rows = []
            for e in manifest['files']:
                path = Path(e['local_path'])
                part = path.with_suffix(path.suffix+'.part')
                # aria2 可写稀疏分段文件，不能用文件逻辑长度冒充下载进度。
                received = path.stat().st_size if path.exists() else 0
                rows.append(dict(path=e['relative_path'], received=received, expected=e['expected_bytes'], complete=path.exists() and received==e['expected_bytes']))
            complete = all(f.done() for f in futures)
            errors = [str(f.exception())[:200] for f in futures if f.done() and f.exception()]
            state = 'failed' if errors else 'downloaded_waiting_user_gpu' if complete else 'downloading'
            save(RUN/'download_status.json', dict(state=state, files=rows, received_bytes=sum(r['received'] for r in rows),
                expected_bytes=manifest['total_download_bytes'], elapsed_seconds=round(time.time()-start),
                errors=errors, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                gpu_inference_started=False, auto_resume=False, received_bytes_definition='completed files only; see per-file aria2 logs for in-progress transfer'))
            if complete:
                for future in futures:
                    future.result()
                break
            time.sleep(5)
    print('DOWNLOADS_COMPLETE_NO_INFERENCE', flush=True)

if __name__ == '__main__':
    main()
