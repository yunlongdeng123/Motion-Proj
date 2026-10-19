"""Read-only CPU snapshot of five run configs, exact code, and frozen road targets."""
from pathlib import Path,PurePosixPath
import hashlib,json,subprocess,tarfile
from collect_native30k_pair import sha
W=Path(__file__).resolve().parent
ROOT=W/'ai_review_source_snapshot'
ARCHIVE=W/'ai_review_source_snapshot.tar.gz'
RECEIPT=W/'ai_review_source_snapshot.json'
REMOTE=r'''
from pathlib import Path
import datetime,gzip,hashlib,io,json,sys,tarfile
BASE=Path('/root/autodl-tmp/drivingeditor_repro')
C=BASE/'campaign/r15_gn11_stage1_v2_20261019'
N=BASE/'campaign/r15_baseline_owner5_20261018'
B=N/'runs/scene0008_089_137_hybrid_30000_seed0_GN11_resume20k'
native=N/'training_source/scene0008_089_137'
items=[];generated={};snapshots={}
def add(p,name):
    assert p.is_file() and p.stat().st_size<2_000_000,(p,'unexpected large review source')
    items.append((p,name))
def core(engine):
    return sorted(p for p in engine.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.yaml','.yml','.md']
      and p.parts[len(engine.parts)] in ['scene','utils','configs','gaussian_renderer','train.py','README.md','LICENSE_gaussian_splatting.md'])
for arm,run,engine in [('baseline',B,native),('full',C/'run30k',C/'engine'),*[(a,C/'ablations'/a/'run30k',C/'ablations'/a/'engine') for a in ['real_only','prior_only','ownership_only']]]:
    status=json.loads((run/'status.json').read_text());assert status['status']=='complete' and status['iteration']==30000
    if arm!='baseline':assert all(status['optimizer_step_counts'][k]==30000 for k in ['static','dynamic','environment'])
    for p in core(engine):add(p,'engines/'+arm+'/'+p.relative_to(engine).as_posix())
    for name in ['effective_config.json','status.json','actual_scene_binding.json','actual_sgcm_binding.json','v2_run_contract.json','v2_training_returned.json']:
        p=run/name
        if p.exists():add(p,'runs/'+arm+'/'+name)
    p=run/'checkpoint_events/iteration_030000.json'
    if p.exists():add(p,'runs/'+arm+'/native30k_event.json')
    p=run/'paired_steps.jsonl'
    if p.exists():
        with p.open('r') as f:lines=[next(f) for _ in range(100)]
        generated['runs/'+arm+'/first100_paired_steps.jsonl']=''.join(lines).encode()
    stats={}
    for n in ['chkpnt30000.pth','dynamic_chkpnt30000.pth','env_light_chkpnt30000.pth','training_state30000.pth']:
        p=run/n;s=p.stat();stats[n]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    snapshots[arm]={'run':str(run),'engine':str(engine),'status':status['status'],'iteration':status['iteration'],
       'optimizer_step_counts':status.get('optimizer_step_counts'),'final_checkpoint_metadata_only':stats}
    if arm!='baseline':
        campaign=C if arm=='full' else C/'ablations'/arm
        mp=campaign/'source_bundle/candidate_manifest.json';manifest=json.loads(mp.read_text())
        for rel,e in manifest['files'].items():assert hashlib.sha256((engine/rel).read_bytes()).hexdigest()==e['candidate_sha256']
        add(mp,'runs/'+arm+'/deployed_manifest.json')
        for name in ['training_launch.json','train_launch.json']:
            if (campaign/name).exists():add(campaign/name,'runs/'+arm+'/'+name)
bundle=C/'input_bundle_002'
add(bundle/'bundle.json','frozen_input/bundle.json')
data=json.loads((bundle/'bundle.json').read_text())
for frame,row in data['frames'].items():
    p=bundle/row['road_target'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['road_target_sha256']
    add(p,'frozen_input/'+row['road_target'])
tools=BASE/'campaign/followup_scene_prep_20261013/tools/followup'
for p in sorted(tools.glob('*.py')):add(p,'binding_tools/'+p.name)
entries=[]
for p,name in items:
    s=p.stat();entries.append({'path':name,'source':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for name,raw in generated.items():entries.append({'path':name,'source':'bounded first100 lines of actual paired_steps.jsonl','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
manifest={'status':'ACTUAL_FIVE_RUN_REVIEW_SOURCE_SNAPSHOT_CPU_READONLY','collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'files':entries,'runs':snapshots,'GPU_used':False,'training_updates':0,'remote_files_written':False,'checkpoint_bytes_downloaded':0,
  'scope':'native baseline engine core and four actual engine cores, configs, native30k events, first100 logs, frozen49 road targets; excludes compiled third-party binaries/full dataset/weights'}
raw=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=5) as z,tarfile.open(fileobj=z,mode='w|') as tar:
    info=tarfile.TarInfo('manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    for (p,name),e in zip(items,entries):
        s=p.stat();assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'];tar.add(p,arcname=name,recursive=False)
    for name,raw in generated.items():
        info=tarfile.TarInfo(name);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
'''

def main():
    if RECEIPT.exists():
        receipt=json.loads(RECEIPT.read_text(encoding='utf-8'));assert sha(ARCHIVE)==receipt['archive_sha256']
    else:
        partial=ARCHIVE.with_suffix('.gz.partial')
        if ARCHIVE.exists() or partial.exists():raise RuntimeError('preserve unmatched review snapshot')
        with partial.open('xb') as f:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','wm-vgpu-1011','/usr/bin/python3','-'],input=REMOTE.encode(),stdout=f,stderr=subprocess.PIPE,timeout=90)
        if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-2000:])
        partial.rename(ARCHIVE);receipt={'archive_sha256':sha(ARCHIVE),'archive_bytes':ARCHIVE.stat().st_size}
        RECEIPT.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    with tarfile.open(ARCHIVE,'r:gz') as t:
        manifest=json.load(t.extractfile('manifest.json'));expected={e['path']:e for e in manifest['files']}
        members=t.getmembers();assert set(m.name for m in members)==set(expected)|{'manifest.json'}
        for m in members:
            rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
            dest=(ROOT/Path(*rel.parts)).resolve();dest.relative_to(ROOT.resolve());raw=t.extractfile(m).read()
            if m.name in expected:assert hashlib.sha256(raw).hexdigest()==expected[m.name]['sha256'] and len(raw)==expected[m.name]['bytes']
            if dest.exists():assert dest.read_bytes()==raw
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    receipt.update(status='READONLY_REVIEW_SOURCE_SNAPSHOT_HASH_VERIFIED',files=len(manifest['files']),GPU_used=False,training_updates=0)
    RECEIPT.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))

if __name__=='__main__':main()
