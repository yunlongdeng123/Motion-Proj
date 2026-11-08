"""Parallel bounded SCP with per-part SHA and complete archive verification."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,json,hashlib,time,gzip
OUT=Path(__file__).parent/'archives'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_v44_20261108_r1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(exist_ok=True);parts=OUT/'parts';parts.mkdir(exist_ok=True);flags=getattr(subprocess,'CREATE_NO_WINDOW',0);subprocess.run(['scp',f'wm-vgpu-1011:{REMOTE}/transfer_manifest.json',str(OUT/'transfer_manifest.json')],check=True,creationflags=flags);manifest=json.loads((OUT/'transfer_manifest.json').read_text());start=time.monotonic()
    def get(row):
        dest=parts/row['name']
        if dest.exists() and dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']:return row['name']
        for attempt in range(3):
            cp=subprocess.run(['scp','-O','-c','aes128-gcm@openssh.com',f'wm-vgpu-1011:{REMOTE}/transfer_parts/{row["name"]}',str(dest)],creationflags=flags,capture_output=True)
            if cp.returncode==0 and dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']:return row['name']
        raise RuntimeError('Verified transfer failed: '+row['name'])
    with ThreadPoolExecutor(max_workers=12) as pool:
        fs=[pool.submit(get,r) for r in manifest['parts']]
        for i,f in enumerate(as_completed(fs),1):print(json.dumps(dict(completed=i,total=len(fs),part=f.result(),seconds=time.monotonic()-start)),flush=True)
    archive=OUT/'recovery_core.tar.gz'
    with archive.open('wb') as f:
        for row in manifest['parts']:
            with (parts/row['name']).open('rb') as q:
                for b in iter(lambda:q.read(8*1024**2),b''):f.write(b)
    assert archive.stat().st_size==manifest['archive_bytes'] and sha(archive)==manifest['archive_sha256']
    h=hashlib.sha256()
    with gzip.open(archive,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    core=json.loads((OUT/'recovery_core_receipt.json').read_text());assert h.hexdigest()==core['sha256']
    receipt=dict(status='PASS_LOCAL_FULL_ARCHIVE_AND_UNCOMPRESSED_SHA',archive=str(archive.resolve()),bytes=archive.stat().st_size,compressed_sha256=manifest['archive_sha256'],uncompressed_sha256=h.hexdigest(),seconds=time.monotonic()-start);(OUT/'local_verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
