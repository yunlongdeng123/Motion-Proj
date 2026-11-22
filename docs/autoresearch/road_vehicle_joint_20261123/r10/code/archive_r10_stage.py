"""R10-only method snapshot. estimate is live; create requires Root's final-stage receipt.

verify only streams the explicitly supplied local tar/JSON: no extraction, model
loading, subprocess, remote-path lookup, GPU access, Git mutation or shutdown.
"""
from pathlib import Path, PurePosixPath
import argparse,datetime,hashlib,json,os,re,stat,tarfile

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUNS=('road_underbody_local_20261123_r10',)
RUN=RUNS[0]
SCHEMA='R10.complete_method_snapshot_archive.v1'
EXCLUDE_DIRS={'archives','report','publication','__pycache__'}
CRITICAL=(
    'code/archive_r10_stage.py','user_execution_contract.json',
    'audit/local_surface_candidate_v2/candidate_geometry.npz',
    'local_texture_seed_v1/texture_seed.npz','local_texture_seed_v1/new_texture_lineage.npz',
    'local_shared_continuous_v1/final_atlas.npz','local_shared_continuous_v1/radiometry_state.npz',
    'local_shared_continuous_v1/solve_state.npz',
    'local_world_observations_v1/086/world_observation.npz','local_world_observations_v1/110/world_observation.npz',
    'new_local_asset_zero_v1/local_field.npz','new_local_asset_zero_v1/manifest.json',
    'measure_v1/receipt.json','new_local_zero_v1/receipt.json','new_local_background_zero_v1/receipt.json',
)


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def pin(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def fingerprint(p):
    s=Path(p).lstat();assert stat.S_ISREG(s.st_mode),'Not a regular source file: '+str(p)
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
EXCLUDE_SUFFIXES={'.pyc','.part','.html','.htm'}
def excluded(rel):return any(x.casefold() in EXCLUDE_DIRS for x in rel.parts) or rel.suffix.casefold() in EXCLUDE_SUFFIXES
def member_name(name):
    assert isinstance(name,str) and name and not any(x in name for x in ['\\',':','\x00'])
    parts=name.split('/');assert not any(x in ('','.','..') for x in parts)
    p=PurePosixPath(name);assert not p.is_absolute() and parts[0]==RUN and len(parts)>1
    assert not excluded(PurePosixPath(*parts[1:])),name
    return name


def inventory(root):
    assert root.is_dir() and root.name==RUN
    files=[];skipped=[]
    for current,dirs,names in os.walk(root,followlinks=False):
        current=Path(current);dirs.sort();names.sort()
        keep=[]
        for name in dirs:
            p=current/name;rel=p.relative_to(root)
            if p.is_symlink():skipped.append(dict(path=rel.as_posix(),reason='symlink_directory'))
            elif name.casefold() not in EXCLUDE_DIRS:keep.append(name)
        dirs[:]=keep
        for name in names:
            p=current/name;rel=p.relative_to(root)
            if excluded(rel):continue
            if p.is_symlink():skipped.append(dict(path=rel.as_posix(),reason='symlink_file'));continue
            if not stat.S_ISREG(p.lstat().st_mode):skipped.append(dict(path=rel.as_posix(),reason='special_file'));continue
            files.append((p,member_name(RUN+'/'+rel.as_posix()),fingerprint(p)))
    return sorted(files,key=lambda x:x[1]),skipped


def estimate(root):
    files,skipped=inventory(root);groups={}
    for path,name,fp in files:
        top=path.relative_to(root).parts[0];row=groups.setdefault(top,dict(files=0,bytes=0));row['files']+=1;row['bytes']+=fp[2]
    n=len(files);total=sum(fp[2] for _,_,fp in files)
    # Payload sum is exact for this live listing. gzip size is not guessed.
    return dict(schema='R10.live_method_payload_estimate.v1',complete=True,created_UTC=now(),run=RUN,
        live_inventory_not_frozen=True,method_stage_frozen=False,overall_run_complete=False,
        eligible_files=n,eligible_payload_bytes=total,eligible_payload_MiB=total/1024**2,eligible_payload_GiB=total/1024**3,
        tar_regular_data_plus_basic_headers_bytes=sum(512+((fp[2]+511)//512)*512 for _,_,fp in files)+1024,
        gzip_bytes='not measured; no archive has been created by estimate',by_top_level=groups,
        excluded_directories=sorted(EXCLUDE_DIRS),excluded_suffixes=sorted(EXCLUDE_SUFFIXES),skipped_nonregular=skipped,
        missing_critical_files=[x for x in CRITICAL if not (root/x).is_file() or (root/x).is_symlink()],
        report_HTML_included=False,GPU_calls=0)


def inside_file(root,value):
    p=Path(value)
    if not p.is_absolute():p=root/p
    resolved=p.resolve();resolved.relative_to(root.resolve())
    assert p.is_file() and not p.is_symlink() and not excluded(p.relative_to(root)),p
    return p


def checked_receipt(root,record):
    assert isinstance(record,dict) and record.get('path') and re.fullmatch('[0-9a-f]{64}',record.get('sha256',''))
    p=inside_file(root,record['path']);assert sha(p)==record['sha256']
    assert read(p).get('complete') is True,'Final required stage not complete: '+str(p)
    return pin(p)


def check_root_signal(root,path):
    x=read(path)
    assert x.get('schema')=='R10.method_snapshot_ready.v1'
    assert x.get('run')==RUN and x.get('complete') is True and x.get('method_stage_frozen') is True
    assert x.get('GPU_stays_on') is True and type(x.get('overall_run_complete')) is bool
    assert type(x.get('whole_scene_restoration_accepted')) is bool
    assert x.get('report_HTML_included') is False
    assert x.get('pending_next_footprint_experiment') is False
    final=checked_receipt(root,x['final_footprint_receipt'])
    stages=[checked_receipt(root,v) for v in x.get('required_complete_receipts',[])]
    if x['overall_run_complete']:
        assert x.get('local_takeback_verified') is True and x.get('report_QA_passed') is True
    return x,dict(final_footprint_receipt=final,required_complete_receipts=stages)


def verify(archive,manifest):
    m=read(manifest);assert m.get('schema')==SCHEMA and m.get('runs')==list(RUNS)
    assert m['archive']['name']==archive.name and archive.stat().st_size==m['archive']['bytes'] and sha(archive)==m['archive']['sha256']
    expected={};total=0
    for row in m['members']:
        name=member_name(row['name']);assert name not in expected
        assert type(row['bytes']) is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row['sha256'])
        expected[name]=row;total+=row['bytes']
    assert total==m['payload_bytes'] and len(expected)==m['member_count']
    seen=set()
    with tarfile.open(archive,'r:gz') as t:
        for entry in t:
            name=member_name(entry.name);assert entry.isfile() and name in expected and name not in seen
            row=expected[name];assert entry.size==row['bytes']
            h=hashlib.sha256();n=0
            with t.extractfile(entry) as f:
                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);n+=len(block)
            assert n==row['bytes'] and h.hexdigest()==row['sha256'],name
            seen.add(name)
    assert seen==set(expected)
    return dict(schema='R10.safe_local_archive_verification.v1',complete=True,decision='PASS',created_UTC=now(),
        archive=pin(archive),manifest=pin(manifest),all_member_SHA_verified=True,member_count=len(seen),payload_bytes=total,
        tar_extracted=False,remote_reference_paths_dereferenced=False,archived_code_executed=False,model_or_NPZ_loaded=False,
        report_HTML_included=False,whole_scene_quality_not_asserted=True,GPU_calls=0)


def create(root,archive,manifest,signal_path,dependencies_path):
    assert not archive.exists() and not manifest.exists()
    for target in [archive,manifest]:
        try:rel=target.resolve().relative_to(root.resolve())
        except ValueError:continue
        assert 'archives' in [x.casefold() for x in rel.parts],'Output must be outside the run, or under excluded archives/'
    signal_pin=pin(signal_path);signal,stagepins=check_root_signal(root,signal_path)
    assert sha(signal_path)==signal_pin['sha256']
    dependency_pin=pin(dependencies_path)
    dependency=read(dependencies_path);assert dependency.get('schema')=='R10.previous_stage_external_dependencies.v1' and dependency.get('complete') is True
    assert dependency.get('large_weights_copied_into_R10') is False
    files,skipped=inventory(root);names={name for _,name,_ in files}
    for value in CRITICAL+tuple(signal.get('critical_members',[])):
        assert member_name(RUN+'/'+value) in names,'Missing required payload: '+value
    rows=[]
    for path,name,fp in files:
        digest=sha(path);assert fingerprint(path)==fp,'Source changed during hash: '+str(path)
        rows.append(dict(name=name,bytes=fp[2],sha256=digest,mtime_ns=fp[3]))
    archive.parent.mkdir(parents=True,exist_ok=True);part=Path(str(archive)+'.part');assert not part.exists()
    try:
        with tarfile.open(part,'w:gz',compresslevel=3,dereference=False) as t:
            for (path,name,fp),row in zip(files,rows):
                assert fingerprint(path)==fp and sha(path)==row['sha256']
                entry=t.gettarinfo(str(path),arcname=name);assert entry.isfile()
                with path.open('rb') as f:
                    fs=os.fstat(f.fileno());assert (fs.st_dev,fs.st_ino,fs.st_size,fs.st_mtime_ns,fs.st_ctime_ns)==fp
                    t.addfile(entry,f)
                assert fingerprint(path)==fp,'Source changed while packing: '+str(path)
        after,_=inventory(root)
        assert [(name,fp) for _,name,fp in after]==[(name,fp) for _,name,fp in files],'Run inventory changed; keep .part and retry from new name after stage freeze'
        assert sha(signal_path)==signal_pin['sha256'] and sha(dependencies_path)==dependency_pin['sha256']
        os.replace(part,archive)
        result=dict(schema=SCHEMA,created_UTC=now(),runs=list(RUNS),archive=dict(name=archive.name,bytes=archive.stat().st_size,sha256=sha(archive)),
            payload_bytes=sum(x['bytes'] for x in rows),member_count=len(rows),members=rows,
            root_delivery=signal_pin,root_delivery_snapshot=signal,validated_stage_receipts=stagepins,
            external_dependency_reference=dependency_pin,external_dependencies=dependency,
            excluded_directories=sorted(EXCLUDE_DIRS),excluded_suffixes=sorted(EXCLUDE_SUFFIXES),skipped_nonregular=skipped,
            original_checkpoints_external_not_copied=True,all_actual_R10_regular_assets_and_failures_included=True,
            report_HTML_included=False,HTML_snapshot_saved_separately_later=True,
            method_stage_frozen=True,overall_run_complete=signal['overall_run_complete'],whole_scene_restoration_accepted=signal['whole_scene_restoration_accepted'],GPU_stays_on=True,GPU_calls=0)
        write(manifest,result)
        verification=verify(archive,manifest);write(Path(str(archive)+'.remote_verify.json'),verification)
        return dict(archive=result['archive'],manifest=pin(manifest),verification=verification)
    except BaseException as exc:
        write(Path(str(archive)+'.failed.json'),dict(complete=False,error_type=type(exc).__name__,message=str(exc),partial_retained=part.exists(),GPU_calls=0))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['estimate','create','verify'])
    p.add_argument('--campaign-root',type=Path,default=CAMPAIGN);p.add_argument('--archive',type=Path);p.add_argument('--manifest',type=Path)
    p.add_argument('--root-delivery',type=Path);p.add_argument('--dependencies',type=Path);p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.mode=='verify':
        assert a.archive and a.manifest;result=verify(a.archive,a.manifest)
    else:
        root=a.campaign_root/RUN
        if a.mode=='estimate':result=estimate(root)
        else:
            assert a.archive and a.manifest and a.root_delivery
            deps=a.dependencies or root/'provenance/previous_stage_archive_reference.json'
            result=create(root,a.archive,a.manifest,a.root_delivery,deps)
    if a.receipt:write(a.receipt,result)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
