"""Verify pending source snapshot and fail-closed final gate; no Git calls."""
from pathlib import Path
import argparse,hashlib,json
import publish_road_vehicle_joint as wrapper


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--selection',type=Path);a=p.parse_args();root=a.root.resolve()
    path=a.selection or max((x for x in (root/'publication').glob('selected_files.pending*.json') if not x.name.endswith('.QA.json')),key=lambda x:x.stat().st_mtime_ns)
    selected=read(path);assert selected['finalized_by_root'] is False and selected['requires_current_stage_review'] is True
    assert not (root/'publication/selected_files.json').exists() and not (root/'publication/narrative.json').exists()
    destinations=[];missing=[]
    for row in selected['files']:
        source=root/row['source'];destinations.append(row['destination'])
        if source.is_file():
            assert source.suffix in {'.py','.js','.json','.md','.svg'} and source.stat().st_size<300000
            assert sha(source)==row['sha256']
        else:missing.append(row['source'])
    assert len(destinations)==len(set(destinations))
    ref=selected['current_stage_snapshot'];status=read(root/ref['source']);assert sha(root/ref['source'])==ref['sha256']
    assert status['snapshot_only'] and status['ready'] is False and status['narrative_generated'] is False
    assert status['human_verdict'] is None and status['problem_solved'] is None and not status['whole_scene_restoration_claimed']
    def forbidden(*args,**kwargs):raise AssertionError('Git prohibited in snapshot QA')
    wrapper.p.git=forbidden;denied=0
    for narrative in [{},dict(current_stage_snapshot_reviewed=True,current_stage_snapshot_sha256='wrong'),dict(current_stage_snapshot_reviewed=True,current_stage_snapshot_sha256=ref['sha256']),
        dict(current_stage_snapshot_reviewed=True,current_stage_snapshot_sha256=ref['sha256'],R10_archive_takeback_reviewed=True)]:
        try:wrapper.current_stage_review(root,narrative,selected)
        except RuntimeError:denied+=1
        else:raise AssertionError('Unfinalized delivery reached the final gate')
    out=dict(schema='R8_R9_R10.publication_snapshot_QA.v1',complete=True,decision='PASS',selection=dict(path=str(path),sha256=sha(path)),files_checked=len(selected['files']),
        missing_final_inputs=missing,unreviewed_final_gate_denials=denied,ready=False,finalized_by_root=False,Git_commands=0,stage_executed=False,commit_created=False,pushed=False)
    target=path.with_name(path.stem+'.QA.json');assert not target.exists();target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':main()
