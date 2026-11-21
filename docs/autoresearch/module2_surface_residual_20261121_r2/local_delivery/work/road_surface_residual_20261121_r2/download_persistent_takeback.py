"""Resumable SFTP takeback with a 30-minute attempt timeout and full file verification."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,os,shutil,subprocess,tarfile,time

DEST=Path(__file__).resolve().parent/'persistent_asset_takeback'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/road_surface_residual_20261121_r2/delivery/persistent_asset_takeback_r2'
DEST.mkdir(parents=True,exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    tmp=Path(p).with_suffix(Path(p).suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    metadata=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','cat '+REMOTE+'/package_receipt.json'],timeout=45)
    rec=json.loads(metadata);assert rec['complete'] and PurePosixPath(rec['archive']).parent.as_posix()==REMOTE
    write(DEST/'remote_package_receipt.json',rec)
    target=DEST/'persistent_assets_and_states.tar.gz';expected=rec['archive_sha256'];expected_bytes=rec['archive_bytes']
    if shutil.disk_usage(DEST).free<expected_bytes+rec['payload_bytes']+32*1024*1024:raise RuntimeError('Insufficient local space for archive and verified extraction')
    batch=DEST/'resume.sftp';batch.write_text('reget "'+rec['archive']+'" "'+target.as_posix()+'"\n',encoding='utf-8')
    for attempt in range(1,5):
        if target.exists() and target.stat().st_size==expected_bytes and sha(target)==expected:break
        if target.exists() and target.stat().st_size>=expected_bytes:
            bad=target.with_name(target.name+'.badsha.'+str(int(time.time())));target.rename(bad)
        before=target.stat().st_size if target.exists()else 0
        write(DEST/'transfer_status.json',dict(state='resuming',attempt=attempt,bytes_before=before,expected_bytes=expected_bytes,timeout_seconds=1800,UTC=now()))
        print(json.dumps(dict(state='resuming',attempt=attempt,bytes_before=before,expected_bytes=expected_bytes)),flush=True)
        with (DEST/f'sftp_attempt{attempt}.log').open('ab')as log:
            proc=subprocess.Popen(['sftp','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','-b',str(batch),'wm-vgpu-1011'],stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic();last=start
            while proc.poll()is None:
                time.sleep(5)
                current=time.monotonic()
                if current-last>=25:
                    size=target.stat().st_size if target.exists()else 0
                    print(json.dumps(dict(state='transferring',bytes=size,expected_bytes=expected_bytes,attempt=attempt)),flush=True);last=current
                    write(DEST/'transfer_status.json',dict(state='transferring',attempt=attempt,bytes=size,expected_bytes=expected_bytes,UTC=now()))
                if current-start>=1800:proc.kill();proc.wait();break
        if target.exists() and target.stat().st_size==expected_bytes and sha(target)==expected:break
        print(json.dumps(dict(state='retry_keep_prefix',attempt=attempt,returncode=proc.returncode)),flush=True)
    assert target.exists()and target.stat().st_size==expected_bytes and sha(target)==expected,'SFTP incomplete; prefix retained for reget'
    print(json.dumps(dict(state='archive_verified',bytes=expected_bytes,SHA256=expected)),flush=True)
    unpacked=DEST/'unpacked';unpacked.mkdir(exist_ok=True)
    with tarfile.open(target,'r:gz')as t:
        for item in t.getmembers():
            rel=PurePosixPath(item.name)
            if rel.is_absolute()or'..'in rel.parts or':'in item.name or item.issym()or item.islnk():raise ValueError('Unsafe archive member '+item.name)
            assert (unpacked/item.name).resolve().is_relative_to(unpacked.resolve())
        t.extractall(unpacked)
    for name,key in [('manifest.json','manifest_sha256'),('dependencies.json','dependencies_sha256'),('checkpoints.json','checkpoints_sha256')]:assert sha(unpacked/name)==rec[key],name
    m=json.loads((unpacked/'manifest.json').read_text());checked=0
    for rel,item in m['files'].items():
        f=unpacked/rel;assert f.is_file()and f.stat().st_size==item['bytes']and sha(f)==item['sha256'],rel;checked+=1
    assert checked==rec['payload_files']
    final=dict(schema='surface_R2_persistent_asset_takeback_verified.v1',complete=True,archive=str(target),archive_bytes=expected_bytes,archive_SHA256=expected,payload_files_verified=checked,payload_bytes=rec['payload_bytes'],manifest_SHA256=rec['manifest_sha256'],dependencies_SHA256=rec['dependencies_sha256'],checkpoints_SHA256=rec['checkpoints_sha256'],unpacked=str(unpacked),standalone_scene_backup=False,new_GPU_operations=0,source_files_unchanged=rec['original_files_unchanged'],UTC=now())
    write(DEST/'takeback_verified.json',final);write(DEST/'transfer_status.json',dict(state='verified',**final));print(json.dumps(final),flush=True)

if __name__=='__main__':main()
