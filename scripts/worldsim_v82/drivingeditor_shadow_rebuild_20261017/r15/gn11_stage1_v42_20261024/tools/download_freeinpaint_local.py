"""Download the fixed inference weight set to an explicit root; no model execution.

V42_WEIGHTS_ROOT, V42_DOWNLOAD_ENDPOINT, and V42_DOWNLOAD_PROXY select location
and transport. Linux defaults to direct access; Windows defaults to local proxy.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(os.environ.get('V42_WEIGHTS_ROOT') or Path(__file__).resolve().parents[2] / 'outputs/V42_Pretrained_Weights_20261024').absolute()
os.environ['V42_WEIGHTS_ROOT'] = str(ROOT)
from download_freeinpaint_weights import REVISIONS, digest, make_manifest

def main():
    entries = make_manifest()
    total = sum(x['size'] for x in entries)
    assert shutil.disk_usage(ROOT).free > total + 8*1024**3
    plan = dict(revisions=REVISIONS, total_bytes=total, files=entries)
    (ROOT / 'download_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    lines = []
    for e in entries:
        path = ROOT / 'hub' / ('models--' + e['repo'].replace('/', '--')) / 'snapshots' / e['revision'] / e['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        e['path'] = str(path)
        if path.exists() and path.stat().st_size==e['size'] and not path.with_name(path.name+'.aria2').exists() and digest(path,e)==e['expected_hash']:
            continue
        lines.extend([os.environ.get('V42_DOWNLOAD_ENDPOINT','https://huggingface.co').rstrip('/') + '/' + e['repo'] + '/resolve/' + e['revision'] + '/' + e['file'],
                      ' dir=' + str(path.parent), ' out=' + path.name])
        if e['algorithm'] == 'sha256':
            lines.append(' checksum=sha-256=' + e['expected_hash'])
    (ROOT / 'aria2_input.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    receipt = dict(status='downloading', pid=os.getpid(), started_at=time.time(), inference_run=False, experiment_started=False,
                   location='local_windows' if os.name=='nt' else 'autodl_remote',
                   remote_transfer='separate_remote_download' if os.name=='nt' else 'not_needed',
                   endpoint=os.environ.get('V42_DOWNLOAD_ENDPOINT','https://huggingface.co'),total_bytes=total, files=[])
    receipt_path = ROOT / 'download_receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2))
    try:
        with (ROOT / 'aria2.log').open('a', encoding='utf-8') as log:
            command = ['aria2c', '--input-file=' + str(ROOT / 'aria2_input.txt'), '--continue=true',
                       '--max-concurrent-downloads=4', '--max-connection-per-server=8', '--split=8', '--min-split-size=16M',
                       '--file-allocation=none', '--check-integrity=true',
                       '--auto-file-renaming=false', '--allow-overwrite=false', '--max-tries=12', '--retry-wait=5',
                       '--connect-timeout=30', '--timeout=60', '--lowest-speed-limit=50K', '--summary-interval=60', '--download-result=hide', '--console-log-level=warn', '--show-console-readout=false']
            proxy=os.environ.get('V42_DOWNLOAD_PROXY','http://127.0.0.1:10808' if os.name=='nt' else '')
            if proxy:
                command.append('--all-proxy='+proxy)
            env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy','no_proxy')}
            process = subprocess.Popen(command, stdout=log, stderr=log,env=env)
            receipt['download_pid'] = process.pid
            receipt_path.write_text(json.dumps(receipt, indent=2))
            returncode = process.wait()
        receipt.update(status='verifying', downloader_returncode=returncode)
        receipt_path.write_text(json.dumps(receipt, indent=2))
        failures=[]
        for e in entries:
            path=Path(e['path'])
            actual=digest(path,e) if path.exists() and path.stat().st_size==e['size'] else None
            verified=actual==e['expected_hash'] and not path.with_name(path.name+'.aria2').exists()
            result=dict(e, actual_hash=actual, verified=verified)
            receipt['files'].append(result)
            if not verified:
                failures.append(e['repo']+'/'+e['file'])
            receipt_path.write_text(json.dumps(receipt,indent=2))
        if failures:
            raise RuntimeError('Incomplete or failed hashes: '+repr(failures))
        for repo,rev in REVISIONS.items():
            refs=ROOT/'hub'/('models--'+repo.replace('/','--'))/'refs'
            refs.mkdir(exist_ok=True)
            (refs/'main').write_text(rev)
        if os.name!='nt':
            alias=ROOT/'hub/models--runwayml--stable-diffusion-inpainting'
            if not alias.exists():
                alias.symlink_to(ROOT/'hub/models--stable-diffusion-v1-5--stable-diffusion-inpainting',target_is_directory=True)
            checkpoints=ROOT/'code/checkpoints'
            if (ROOT/'code').is_dir() and not checkpoints.exists():
                checkpoints.symlink_to(ROOT/'hub/models--jyproris--FreeInpaint/snapshots'/REVISIONS['jyproris/FreeInpaint']/'checkpoints',target_is_directory=True)
        receipt.update(status='complete',completed_at=time.time(),file_count=len(entries))
    except Exception as e:
        receipt.update(status='failed',error=repr(e))
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))

if __name__=='__main__':
    main()
