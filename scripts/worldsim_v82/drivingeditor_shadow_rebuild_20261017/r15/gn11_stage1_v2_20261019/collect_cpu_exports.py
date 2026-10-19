"""Export existing images after no-GPU boot; remote files are read only."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, subprocess, tarfile
from collect_native30k_pair import sha

W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
ARCHIVE=W/'cpu_exports.tar.gz'
DEST=W/'cpu_exports'
RECEIPT=W/'cpu_exports_download.json'
REMOTE=r'''
from pathlib import Path
import datetime,gzip,hashlib,io,json,struct,sys,tarfile
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019')
entries=[];sources={};files=[];states={}
ids=['0008%03d'%i for i in range(89,138)]
for arm in ['full','real_only','prior_only','ownership_only']:
    campaign=C if arm=='full' else C/'ablations'/arm
    run=campaign/'run30k';status=json.loads((run/'status.json').read_text())
    assert status['status']=='complete' and status['iteration']==30000
    assert all(status['optimizer_step_counts'][k]==30000 for k in ['static','dynamic','environment'])
    out=campaign/'full49'
    states[arm]={'training_status':status['status'],'iteration':status['iteration'],'optimizer_step_counts':status['optimizer_step_counts'],
        'independent_PNG_count':len(list(out.glob('*/*.png'))),'independent_receipt_exists':(out/'receipt.json').exists()}
    if arm in ['prior_only','ownership_only']:
        ep=run/'checkpoint_events/iteration_030000.json';event=json.loads(ep.read_text())
        assert event['status']=='complete' and event['iteration']==30000
        rows=sorted(event['frames'],key=lambda r:r['image_name'])
        assert [r['image_name'] for r in rows]==ids
        sources[arm]={'kind':'existing_native_checkpoint_evaluation','event_path':str(ep),'event_sha256':hashlib.sha256(ep.read_bytes()).hexdigest()}
        for row in rows:
            assert set(row['channels'])=={'gt','composite','static','dynamic'}
            for channel,src in row['channels'].items():
                p=Path(src).resolve();p.relative_to(run/'eval/train_30000_render')
                files.append((p,'native/'+arm+'/'+channel+'/'+row['image_name']+'.png'))
    else:
        receipt=json.loads((out/'receipt.json').read_text());restore=json.loads((out/'restore_receipt.json').read_text())
        progress=json.loads((out/'progress.json').read_text())
        assert receipt['frames']==49 and receipt['parameter_updates']==0 and receipt['source_checkpoints_unchanged']
        assert sorted(r['frame_id'] for r in receipt['records'])==ids
        assert restore['source_run']==str(run) and restore['model_and_Adam_values_exact'] and restore['parameter_updates']==0
        assert progress['frames']==49 and progress['parameter_updates']==0
        assert receipt['channels']==['gt','full','dynamic','static','delete_all','dynamic_alpha','static_alpha']
        for component,expected in receipt['source_components'].items():
            s=(run/component).stat();assert s.st_size==expected['bytes'] and s.st_mtime_ns==expected['mtime_ns']
        sources[arm]={'kind':'completed_independent_zero_update49','receipt':receipt,'restore_receipt':restore,'progress':progress}
        for frame in ids:
            for channel in receipt['channels']:files.append((out/frame/(channel+'.png'),'independent/'+arm+'/'+frame+'/'+channel+'.png'))
        for name in ['receipt.json','restore_receipt.json','progress.json']:files.append((out/name,'independent/'+arm+'/'+name))
for p,name in files:
    raw=p.read_bytes();s=p.stat()
    if name.endswith('.png'):
        assert raw[:8]==b'\x89PNG\r\n\x1a\n' and struct.unpack('>II',raw[16:24])==(480,320)
    entries.append({'path':name,'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':s.st_mtime_ns})
md={}
for key in ['array_state','degraded','sync_action']:
    p=Path('/sys/block/md0/md')/key;md[key]=p.read_text().strip() if p.exists() else None
manifest={'status':'EXISTING_CPU_ONLY_EXPORT_WITH_EXPLICIT_INDEPENDENT_SCOPE','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'sources':sources,'states':states,'files':entries,'PNG_count':sum(x['path'].endswith('.png') for x in entries),'full_frame_wh':[480,320],
    'GPU_used_for_this_export':False,'training_updates':0,'remote_files_written':False,'md0':md,
    'compute_device_nodes':[str(p) for p in Path('/dev').glob('nvidia*')],
    'independent_completed':['full','real_only'],'independent_pending':['prior_only','ownership_only'],
    'interruption':'prior_only independent renderer SSH disconnected during user no-GPU reboot; no receipt or PNG; ownership renderer not started',
    'quality_accepted':False,'human_verdict':None}
raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as z,tarfile.open(fileobj=z,mode='w|') as tar:
    info=tarfile.TarInfo('manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    for (p,name),e in zip(files,entries):
        s=p.stat();assert (s.st_size,s.st_mtime_ns)==(e['bytes'],e['mtime_ns'])
        tar.add(p,arcname=name,recursive=False)
        s=p.stat();assert (s.st_size,s.st_mtime_ns)==(e['bytes'],e['mtime_ns'])
'''

def main():
    if RECEIPT.exists():
        r=json.loads(RECEIPT.read_text(encoding='utf-8'))
        assert ARCHIVE.exists() and sha(ARCHIVE)==r['archive_sha256']
    else:
        partial=ARCHIVE.with_suffix('.gz.partial')
        if ARCHIVE.exists() or partial.exists():raise RuntimeError('preserve unreceipted archive or partial')
        with partial.open('xb') as f:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],input=REMOTE.encode(),stdout=f,stderr=subprocess.PIPE,timeout=300)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-4000:])
        partial.rename(ARCHIVE)
        r={'status':'CPU_ONLY_IMAGES_DOWNLOADED','archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size,
           'downloaded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_files_written':False,'GPU_used':False,'training_updates':0}
        RECEIPT.write_text(json.dumps(r,indent=2),encoding='utf-8')
    with tarfile.open(ARCHIVE,'r:gz') as tar:
        manifest=json.load(tar.extractfile('manifest.json'));expected={e['path']:e for e in manifest['files']}
        members=tar.getmembers();assert len(expected)==1084 and manifest['PNG_count']==1078
        assert set(m.name for m in members)==set(expected)|{'manifest.json'}
        for m in members:
            rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
            dest=(DEST/Path(*rel.parts)).resolve();dest.relative_to(DEST.resolve());raw=tar.extractfile(m).read()
            if m.name in expected:assert len(raw)==expected[m.name]['bytes'] and hashlib.sha256(raw).hexdigest()==expected[m.name]['sha256']
            if dest.exists():assert dest.read_bytes()==raw
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    # Existing pair is deliberately reused, not downloaded again.
    original=json.loads((W/'native30k_pair/manifest.json').read_text(encoding='utf-8'))
    for arm in ['prior_only','ownership_only']:
        for frame in range(89,138):
            name='0008%03d.png'%frame
            assert sha(DEST/'native'/arm/'gt'/name)==sha(W/'native30k_pair/full/gt'/name)
    import numpy as np
    from PIL import Image
    quantization={}
    for arm in ['full','real_only']:
        errors=[]
        for frame in range(89,138):
            p=DEST/'independent'/arm/('0008%03d'%frame)/'gt.png'
            assert sha(p)==sha(DEST/'independent/full'/('0008%03d'%frame)/'gt.png')
            a=np.asarray(Image.open(p).convert('RGB'),dtype=np.int16)
            b=np.asarray(Image.open(W/'native30k_pair'/arm/'gt'/('0008%03d.png'%frame)).convert('RGB'),dtype=np.int16)
            delta=np.abs(a-b);assert delta.max()<=1
            errors.append({'frame':frame,'max_abs_255':int(delta.max()),'RGB_MAE_255':float(delta.mean())})
        quantization[arm]=errors
    r.update(PNG_count=1078,all_member_hashes_verified=True,matching_49_frame_IDs=True,
        native_four_group_GT_hashes_equal=True,independent_pair_GT_hashes_equal=True,
        native_vs_independent_GT_quantization=quantization,
        independent_completed=manifest['independent_completed'],independent_pending=manifest['independent_pending'])
    RECEIPT.write_text(json.dumps(r,indent=2),encoding='utf-8')
    (O/'cpu_export_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (O/'cpu_export_download.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    for arm in ['full','real_only']:
        for key in ['receipt','restore_receipt','progress']:
            d=O/'independent_receipts'/arm;d.mkdir(parents=True,exist_ok=True)
            (d/(key+'.json')).write_bytes((DEST/'independent'/arm/(key+'.json')).read_bytes())
    print(json.dumps(r))

if __name__=='__main__':main()
