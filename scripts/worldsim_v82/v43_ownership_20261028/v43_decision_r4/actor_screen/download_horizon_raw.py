"""并行取回已逐像素定位的十个完整Waymo片段；断点文件保留，成功后原子落位。"""
import concurrent.futures as futures
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp/codex_tmp/v43_r4_actor_screen_20261031')
DEST = Path('/root/autodl-tmp/datasets/waymo_v1_4_3/individual_files/validation')

def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def download(row):
    start = time.monotonic()
    target = DEST / row['tfrecord_name']
    expected = row['full_raw_bytes']
    result = dict(scene_number=row['scene_number'], context_id=row['context_id'], path=str(target), expected_bytes=expected)
    if target.exists():
        result.update(status='existing_exact_size' if target.stat().st_size == expected else 'existing_size_mismatch', actual_bytes=target.stat().st_size)
    else:
        partial = target.with_suffix(target.suffix + '.r4_download.part')
        log = ROOT / ('download_' + row['scene_number'] + '.log')
        command = ['curl', '--location', '--fail', '--retry', '3', '--retry-delay', '2', '--connect-timeout', '20', '--speed-time', '60', '--speed-limit', '1024', '--max-time', '1200', '--continue-at', '-', '--output', str(partial), '--write-out', '\nHTTP=%{http_code} downloaded=%{size_download} speed=%{speed_download}\n', row['tested_fixed_mirror_url'] + '?download=true']
        result.update(source_uri=row['tested_fixed_mirror_url'], partial_path=str(partial), command=command)
        with log.open('w') as stream:
            process = subprocess.run(command, stdout=stream, stderr=stream)
        result.update(exit_code=process.returncode, downloaded_bytes=partial.stat().st_size if partial.exists() else 0, log=str(log))
        if process.returncode == 0 and partial.stat().st_size == expected:
            partial.replace(target)
            result.update(status='download_complete_size_verified_crc_pending', actual_bytes=expected)
        else:
            result['status'] = 'download_incomplete_preserved'
    result['wall_seconds'] = time.monotonic() - start
    write(ROOT / ('download_' + row['scene_number'] + '.json'), result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT / 'horizon_source_registry.json').read_text(encoding='utf-8-sig'))
    began = time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, registry['rows']))
    write(ROOT / 'download_summary.json', dict(task='WS-V43-R4-ACTOR-SCREEN-20261031', requested_scenes=11, parallel_downloads=3, results=results, wall_seconds=time.monotonic()-began, downloads_are_not_reconstruction=True))
