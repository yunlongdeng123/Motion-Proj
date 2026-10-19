"""下载两组已存在的30K原生49评估PNG；远端仅只读并向stdout打包。"""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,subprocess,tarfile

W=Path(__file__).resolve().parent
ARCHIVE=W/'native30k_pair.tar.gz'
RECEIPT=W/'native30k_pair_download.json'
DEST=W/'native30k_pair'

REMOTE=r'''
from pathlib import Path
import datetime,gzip,hashlib,io,json,struct,sys,tarfile
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019')
entries=[];events={};frames={};files=[]
for arm,campaign in [('full',C),('real_only',C/'ablations/real_only')]:
    run=campaign/'run30k';ep=run/'checkpoint_events/iteration_030000.json'
    event=json.loads(ep.read_text());status=json.loads((run/'status.json').read_text())
    assert event['status']=='complete' and event['iteration']==30000
    assert status['status']=='complete' and status['iteration']==30000
    assert all(status['optimizer_step_counts'][k]==30000 for k in ['static','dynamic','environment'])
    rows=sorted(event['frames'],key=lambda r:r['image_name'])
    assert [r['frame_id'] for r in rows]==['image_0/0008%03d'%i for i in range(89,138)]
    events[arm]={'path':str(ep),'sha256':hashlib.sha256(ep.read_bytes()).hexdigest(),
        'optimizer_step_counts':status['optimizer_step_counts'],'iteration':30000}
    frames[arm]=[{k:r[k] for k in ['frame_id','timestamp','width','height']} for r in rows]
    for row in rows:
        assert (row['width'],row['height'])==(480,320)
        assert set(row['channels'])=={'gt','composite','static','dynamic'}
        for channel,src in row['channels'].items():
            p=Path(src).resolve();p.relative_to(run/'eval/train_30000_render')
            raw=p.read_bytes();assert raw[:8]==b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II',raw[16:24])==(480,320)
            stat=p.stat();name=arm+'/'+channel+'/'+row['image_name']+'.png'
            entries.append({'path':name,'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'mtime_ns':stat.st_mtime_ns,'frame_id':row['frame_id'],'channel':channel})
            files.append((p,name,stat.st_size,stat.st_mtime_ns))
assert frames['full']==frames['real_only']
lookup={e['path']:e for e in entries}
for i in range(89,138):
    assert lookup['full/gt/0008%03d.png'%i]['sha256']==lookup['real_only/gt/0008%03d.png'%i]['sha256']
manifest={'status':'EXISTING_NATIVE_30K_49_FRAME_CHANNEL_EXPORT','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'events':events,'frames':frames,'files':entries,'PNG_count':len(entries),'full_frame_wh':[480,320],
    'channels':['gt','composite','static','dynamic'],'GPU_used':False,'training_updates':0,
    'independent_delete_evaluation':False,'quality_accepted':False}
raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|') as tar:
    info=tarfile.TarInfo('manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    for p,name,size,mtime in files:
        s=p.stat();assert (s.st_size,s.st_mtime_ns)==(size,mtime)
        tar.add(p,arcname=name,recursive=False)
        s=p.stat();assert (s.st_size,s.st_mtime_ns)==(size,mtime)
'''

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()

def main():
    if RECEIPT.exists():
        receipt=json.loads(RECEIPT.read_text(encoding='utf-8'))
        if not ARCHIVE.exists() or sha(ARCHIVE)!=receipt['archive_sha256']:
            raise RuntimeError('existing receipt/archive differs; preserve for diagnosis')
    else:
        partial=ARCHIVE.with_suffix(ARCHIVE.suffix+'.partial')
        if ARCHIVE.exists() or partial.exists():raise RuntimeError('unreceipted archive/partial exists; preserve for diagnosis')
        with partial.open('xb') as f:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],
                input=REMOTE.encode(),stdout=f,stderr=subprocess.PIPE,timeout=180)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-4000:])
        partial.rename(ARCHIVE)
        receipt={'status':'DOWNLOADED_EXISTING_NATIVE_PNGS_ONLY','archive_sha256':sha(ARCHIVE),
            'archive_bytes':ARCHIVE.stat().st_size,'downloaded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'remote_files_written':False,'GPU_used':False,'training_updates':0}
        RECEIPT.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    with tarfile.open(ARCHIVE,'r:gz') as tar:
        members=tar.getmembers();manifest=json.load(tar.extractfile('manifest.json'))
        expected={v['path']:v for v in manifest['files']}
        assert len(expected)==392 and set(x.name for x in members)==set(expected)|{'manifest.json'}
        for member in members:
            rel=PurePosixPath(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe archive member')
            dest=(DEST/Path(*rel.parts)).resolve();dest.relative_to(DEST.resolve())
            raw=tar.extractfile(member).read()
            if member.name in expected:
                e=expected[member.name];assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['sha256']
            if dest.exists():assert dest.read_bytes()==raw
            else:
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    receipt.update(PNG_count=392,all_file_hashes_verified=True,matching_49_frame_IDs_and_GT=True)
    RECEIPT.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
