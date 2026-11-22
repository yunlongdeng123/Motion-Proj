"""Write a new pending selection updating only three final entries, preserving its reviewed stage snapshot."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,copy,hashlib,json

FINAL={'README.md':'README.md','compact_result.json':'compact_result.json','final_delivery.json':'publication/final_delivery.json'}
ARCHITECTURE=('architecture_components.svg','publication/architecture_components.svg')
LIMIT=300000
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inside(root,name):
    rel=PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:raise ValueError('Expected a safe relative run path')
    p=(root/Path(*rel.parts)).resolve()
    if not p.is_relative_to(root):raise ValueError('Path escapes this R9 run')
    return p
def write_new(p,d):
    if p.exists():raise FileExistsError('Preserve the reviewed selection: '+str(p))
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def refresh(root,selection,output=None,refresh_architecture=False):
    root=Path(root).resolve();pub=(root/'publication').resolve();selection=Path(selection).resolve()
    if root.name=='road_underbody_local_20261123_r10' or not selection.is_relative_to(pub):raise ValueError('Only the external R9 publication directory is allowed')
    before=read(selection);input_sha=sha(selection)
    assert before['run_id']==root.name and before['finalized_by_root'] is False
    assert before['requires_current_stage_review'] is True
    stage=before['current_stage_snapshot'];stage_path=inside(root,stage['source'])
    assert sha(stage_path)==stage['sha256'],'The reviewed stage snapshot changed'
    status=read(stage_path)
    assert status['R10_full_archive_takeback_verified'] is True,'Prepare the single reviewed snapshot after actual R10 archive takeback'
    wanted=dict(FINAL)
    if refresh_architecture:wanted[ARCHITECTURE[0]]=ARCHITECTURE[1]
    after=copy.deepcopy(before);indices={}
    for i,row in enumerate(before['files']):
        dest=row['destination'];assert dest not in indices,'Duplicate selected destination';indices[dest]=i
    refreshed=[]
    for dest,source in wanted.items():
        if dest not in indices:raise ValueError('Missing existing final-entry slot: '+dest)
        i=indices[dest];old=before['files'][i];assert old['source']==source
        p=inside(root,source);assert p.is_file() and p.stat().st_size<LIMIT
        new=copy.deepcopy(old);new.update(exists_at_preparation=True,sha256=sha(p),bytes=p.stat().st_size)
        after['files'][i]=new
        refreshed.append(dict(source=source,destination=dest,old_sha256=old.get('sha256'),new_sha256=new['sha256'],bytes=new['bytes']))
    delivery=read(inside(root,FINAL['final_delivery.json']))
    assert delivery.get('ready') is True and delivery.get('finalized_by_root') is True
    assert delivery['current_stage_snapshot_sha256']==stage['sha256'],'Final delivery binds another reviewed snapshot'
    assert isinstance(delivery.get('delivery_scope'),str) and delivery['delivery_scope'].strip()
    assert type(delivery.get('whole_scene_restoration_accepted')) is bool
    compact=read(inside(root,FINAL['compact_result.json']))
    assert 'human_verdict' in compact and compact['human_verdict'] is None
    assert 'problem_solved' in compact and compact['problem_solved'] is None
    unchanged=0
    for i,(old,new) in enumerate(zip(before['files'],after['files'])):
        if old['destination'] in wanted:continue
        assert old==new
        p=inside(root,old['source'])
        assert p.is_file() and p.stat().st_size<LIMIT and sha(p)==old.get('sha256'),'Non-final input changed; it is not silently refreshed: '+old['source']
        if old.get('bytes') is not None:assert p.stat().st_size==old['bytes']
        unchanged+=1
    assert {k:v for k,v in before.items() if k!='files'}=={k:v for k,v in after.items() if k!='files'}
    assert after['current_stage_snapshot']==before['current_stage_snapshot'] and after['finalized_by_root'] is False
    assert sha(selection)==input_sha and sha(stage_path)==stage['sha256']
    output=Path(output).resolve() if output else pub/('selected_files.pending.refresh.'+datetime.now(timezone.utc).strftime('%y%m%dT%H%M%S%f')+'.json')
    if not output.is_relative_to(pub) or output==selection:raise ValueError('New selection must remain under R9/publication and preserve its input')
    assert output.suffix=='.json' and output.parent.is_dir()
    write_new(output,after)
    receipt=dict(schema='publication.final_entry_refresh_CPU.v1',complete=True,decision='PASS',input_selection=dict(path=str(selection),sha256=input_sha),
        output_selection=dict(path=str(output),sha256=sha(output)),preserved_current_stage_snapshot=stage,
        refreshed_entries=refreshed,architecture_refreshed_only_by_explicit_option=refresh_architecture,
        other_rows_verified_unchanged=unchanged,all_other_selection_fields_unchanged=True,input_manifest_and_stage_file_unchanged=True,
        finalized_by_root=False,narrative_read_or_written=False,status_snapshot_generated=False,R10_paths_dereferenced=False,
        Git_commands=0,stage_executed=False,commit_created=False,pushed=False)
    # Short physical name also keeps Windows paths within normal limits, and
    # avoids matching selected_files.pending*.json in selection discovery.
    receipt_id=sha(output)[:8]+'_'+hashlib.sha256(output.name.encode('utf-8')).hexdigest()[:8]
    rp=pub/('refresh_receipt_'+receipt_id+'.json');write_new(rp,receipt)
    return receipt
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--refresh-architecture',action='store_true');a=p.parse_args()
    print(json.dumps(refresh(a.root,a.selection,a.output,a.refresh_architecture),ensure_ascii=False))
if __name__=='__main__':main()
