"""Small CPU controls for final-entry refresh without snapshot regeneration or mutable file-pin drift."""
from pathlib import Path
import argparse,json,tempfile
from refresh_final_selection_cpu import refresh,sha,write_new

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    base_parent=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='rq_',dir=base_parent) as tmp:
        base=Path(tmp);assert base.resolve().is_relative_to(base_parent)
        r=base/'r';pub=r/'publication';pub.mkdir(parents=True)
        stage=pub/'stage.json';write_new(stage,dict(snapshot_only=True,ready=False,R10_full_archive_takeback_verified=True))
        other=pub/'fixed.py';other.write_text('x=1\n');arch=pub/'architecture_components.svg';arch.write_text('<svg/>\n')
        def row(source,dest):
            q=r/source;return dict(source=source,destination=dest,require_ready=dest=='final_delivery.json',exists_at_preparation=q.exists(),sha256=sha(q) if q.exists() else None,bytes=q.stat().st_size if q.exists() else None)
        selected=dict(run_id=r.name,finalized_by_root=False,requires_current_stage_review=True,current_stage_snapshot=dict(source='publication/stage.json',sha256=sha(stage)),
            files=[row('README.md','README.md'),row('compact_result.json','compact_result.json'),row('publication/final_delivery.json','final_delivery.json'),row('publication/fixed.py','code/fixed.py'),row('publication/architecture_components.svg','architecture_components.svg')])
        source=pub/'selected_files.pending.json';write_new(source,selected);old_sha=sha(source)
        (r/'README.md').write_text('# Final navigation\n');write_new(r/'compact_result.json',dict(human_verdict=None,problem_solved=None,whole_scene_restoration_accepted=False))
        delivery=dict(ready=True,finalized_by_root=True,current_stage_snapshot_sha256=sha(stage),delivery_scope='negative diagnostic phase',whole_scene_restoration_accepted=False)
        write_new(pub/'final_delivery.json',delivery)
        result=refresh(r,source,pub/'new.pending.json');new=json.loads((pub/'new.pending.json').read_text())
        assert sha(source)==old_sha and new['current_stage_snapshot']==selected['current_stage_snapshot']
        assert new['files'][3:]==selected['files'][3:] and new['finalized_by_root'] is False
        assert not (pub/'narrative.json').exists() and result['other_rows_verified_unchanged']==2
        rejected=0
        other.write_text('x=2\n')
        try:refresh(r,source,pub/'bad_fixed.pending.json')
        except AssertionError:rejected+=1
        else:raise AssertionError('Changed non-final source was silently refreshed')
        other.write_text('x=1\n');wrong=dict(delivery,current_stage_snapshot_sha256='wrong');(pub/'final_delivery.json').write_text(json.dumps(wrong))
        try:refresh(r,source,pub/'bad_stage.pending.json')
        except AssertionError:rejected+=1
        else:raise AssertionError('Wrong snapshot delivery passed')
        (pub/'final_delivery.json').write_text(json.dumps(delivery));arch.write_text('<svg><g/></svg>\n')
        try:refresh(r,source,pub/'bad_arch.pending.json')
        except AssertionError:rejected+=1
        else:raise AssertionError('Architecture changed without the explicit option')
        explicit=refresh(r,source,pub/'explicit_arch.pending.json',True);assert len(explicit['refreshed_entries'])==4
    out=dict(schema='publication.final_entry_refresh_CPU_QA.v1',complete=True,decision='PASS',successful_refresh_controls=2,rejected_tamper_controls=rejected,
        three_entries_only_by_default=True,current_stage_snapshot_unchanged=True,other_filepins_unchanged=True,input_manifest_preserved=True,
        explicit_architecture_supported=True,no_ready_or_narrative_or_status_created=True,finalized_by_root_remains_false=True,R10_access=0,Git_commands=0)
    write_new(a.output,out);print(json.dumps(out))
if __name__=='__main__':main()
