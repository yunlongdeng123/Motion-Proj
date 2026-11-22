"""Freeze and verify the complete two-stage method payload, excluding archives.

No model loading, GPU execution, report creation, Git mutation or shutdown.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, tarfile

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUNS = ('road_vehicle_joint_dggt_20261123_r8', 'old_raw_teacher_joint_20261123_r9')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def verify(archive,manifest):
    m=json.loads(manifest.read_text(encoding='utf-8'))
    assert archive.stat().st_size==m['archive']['bytes'] and sha(archive)==m['archive']['sha256']
    expected={r['name']:r for r in m['members']};seen=set()
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            assert member.isfile() and member.name in expected and member.name not in seen
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            h=hashlib.sha256();n=0
            with t.extractfile(member) as f:
                for b in iter(lambda:f.read(8*1024**2),b''): h.update(b);n+=len(b)
            assert n==expected[member.name]['bytes'] and h.hexdigest()==expected[member.name]['sha256'],member.name
            seen.add(member.name)
    assert seen==set(expected)
    return dict(complete=True,archive_SHA_verified=True,every_member_SHA_verified=True,members=len(seen),HTML_included=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['create','verify']);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root-delivery',type=Path);a=p.parse_args()
    if a.mode=='verify':print(json.dumps(verify(a.archive,a.manifest)));return
    assert a.root_delivery and a.root_delivery.is_file(),'Root final-stage receipt required'
    root=json.loads(a.root_delivery.read_text());assert root['complete'] is True
    assert root['method_stage_frozen'] is True and root['GPU_stays_on'] is True
    # A completed method-stage snapshot may precede full archive takeback and
    # continuing work. It is never silently treated as final task completion.
    assert root['overall_run_complete'] is False or (root['local_takeback_verified'] and root['report_QA_passed'])
    assert not a.archive.exists() and not a.manifest.exists()
    files=[]
    for name in RUNS:
        run=CAMPAIGN/name
        for path in sorted(run.rglob('*')):
            rel=path.relative_to(run)
            if any(s in ('archives','publication','report','__pycache__') for s in rel.parts):continue
            if not path.is_file() or path.is_symlink() or path.suffix in ('.pyc','.part'):continue
            files.append(path)
    members=[dict(name=p.relative_to(CAMPAIGN).as_posix(),bytes=p.stat().st_size,sha256=sha(p),mtime_ns=p.stat().st_mtime_ns) for p in files]
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.archive,'w:gz',compresslevel=3,dereference=True) as t:
        for path,row in zip(files,members):
            assert path.stat().st_mtime_ns==row['mtime_ns'] and sha(path)==row['sha256']
            t.add(path,arcname=row['name'],recursive=False)
            assert path.stat().st_mtime_ns==row['mtime_ns'],'Source changed during freeze'
    value=dict(schema='R8_R9.complete_method_archive.v1',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        archive=dict(name=a.archive.name,bytes=a.archive.stat().st_size,sha256=sha(a.archive)),
        root_delivery=dict(path=str(a.root_delivery),sha256=sha(a.root_delivery)),members=members,
        excluded=['archives','publication','report (the shared HTML/report code continues with later stages)','__pycache__','*.pyc','*.part','symlinks'],
        external_dependencies='All original data, checkpoints and parent native/A20/road assets remain external and SHA-referenced by the preserved contracts; new persistent fitted assets are included.',
        HTML_included=False,GPU_calls=0,GPU_stays_on=True,overall_run_complete=root['overall_run_complete'])
    write(a.manifest,value)
    result=verify(a.archive,a.manifest);write(a.archive.with_suffix(a.archive.suffix+'.verify.json'),result)
    print(json.dumps(dict(archive=value['archive'],verification=result)))


if __name__=='__main__':main()
