"""Pinned checkpoint transfer with verified HTTP ranges and finite stall time.

Existing prefix and previous logs are retained. No model deserialization/GPU.
"""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, re, shutil, subprocess, time

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_vehicle_joint_dggt_20261123_r8')
META = json.loads(Path('/root/autodl-tmp/dggt/.preparation/assets_core.json').read_text())['assets'][0]
TARGET = Path(META['target']); PART = TARGET.with_name(TARGET.name + '.part')
STATE = ROOT / 'runtime/core_ranges_restore.json'
EVENTS = ROOT / 'runtime/core_ranges_restore.events.jsonl'
CHUNK = 64 * 1024**2


def log(event, **extra):
    row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event, **extra)
    with EVENTS.open('a') as f: f.write(json.dumps(row) + '\n')
    STATE.write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row), flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def main():
    lock = open(STATE.parent / 'transfer.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected = META['expected_bytes']
    if TARGET.exists():
        assert TARGET.stat().st_size == expected and digest(TARGET) == META['sha256']
        log('verified_complete', bytes=expected, sha256=META['sha256'], reused=True)
        return
    failures = 0; number = 0
    while not PART.exists() or PART.stat().st_size < expected:
        offset = PART.stat().st_size if PART.exists() else 0
        assert 0 <= offset < expected
        assert shutil.disk_usage(TARGET.parent).free - (expected-offset) > 25 * 1024**3
        end = min(expected-1, offset+CHUNK-1)
        number += 1
        stem = ROOT / f'runtime/range_{offset}_{number}'
        body = stem.with_suffix('.body'); header = stem.with_suffix('.headers')
        # Retry is our outer loop, so curl never resets or truncates PART.
        mirror = failures % 2 == 0
        url = META['official_url'].replace('huggingface.co', 'hf-mirror.com') if mirror else META['official_url']
        url += '?download=true&r9range=' + str(offset)
        cmd = ['curl','--silent','--show-error','--location','--fail','--retry','0',
               '--connect-timeout','15','--max-time','150','--speed-time','30','--speed-limit','16384',
               '--range',f'{offset}-{end}','--dump-header',str(header),'--output',str(body),url]
        if mirror: cmd[1:1] = ['--noproxy','*']
        log('range_start', offset=offset, end=end, mirror=mirror)
        started = time.monotonic()
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        text = header.read_text(errors='replace') if header.exists() else ''
        ranges = re.findall(r'(?im)^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$', text)
        statuses = re.findall(r'(?m)^HTTP/\S+\s+(\d+)', text)
        count = body.stat().st_size if body.exists() else 0
        valid = (statuses and statuses[-1]=='206' and ranges and
                 tuple(map(int,ranges[-1]))==(offset,end,expected) and 0<count<=end-offset+1)
        if valid:
            assert (PART.stat().st_size if PART.exists() else 0) == offset
            with PART.open('ab') as dst, body.open('rb') as src:
                shutil.copyfileobj(src,dst,1024**2); dst.flush(); os.fsync(dst.fileno())
            body.unlink(); header.unlink()
            failures = 0
            log('progress', bytes=offset+count, expected_bytes=expected, accepted_range_prefix=count,
                curl_exit=proc.returncode, seconds=time.monotonic()-started, mirror=mirror)
        else:
            failures += 1
            # Preserve small headers and error codes; signed URLs are never printed.
            log('range_rejected', offset=offset, bytes=count, curl_exit=proc.returncode,
                final_status=statuses[-1] if statuses else None, failures=failures)
            if failures >= 8: raise RuntimeError('Eight consecutive range requests failed; prefix preserved')
            time.sleep(min(failures*2,10))
    actual = digest(PART)
    log('hash_checked', bytes=PART.stat().st_size, sha256=actual, expected_sha256=META['sha256'])
    assert PART.stat().st_size == expected and actual == META['sha256'], 'Preserved, not promoted'
    os.link(PART,TARGET); PART.unlink()
    log('verified_complete', bytes=expected, sha256=actual, reused=False)


if __name__ == '__main__': main()
