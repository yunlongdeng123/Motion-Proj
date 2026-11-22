"""Explicit root-only finalization of the prepared method-stage signal; default is dry-run."""
from pathlib import Path
import argparse,json
import archive_r10_stage as archive

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--root-review',type=Path);p.add_argument('--report-qa',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args();r=a.root
    draft_path=r/'docs/final_method_snapshot_receipt.final_pending.json';d=archive.read(draft_path)
    assert d['complete'] is False and d['method_stage_frozen'] is False and d['no_further_experiments_or_training']
    for ref in d['required_complete_receipts']:archive.checked_receipt(r,ref)
    for ref in d['retained_incomplete_failure_receipts']:
        q=archive.inside_file(r,ref['path']);assert archive.sha(q)==ref['sha256'] and archive.read(q)['complete'] is False
    assert archive.sha(d['resources']['path'])==d['resources']['sha256']
    for rel in d['critical_members']:assert archive.inside_file(r,rel).is_file()
    if not a.execute:
        print(json.dumps(dict(preflight=True,method_stage_frozen=False,freeze_signal_written=False,required_root_inputs=['--root-review actual all-five review','--report-qa passed actual QA JSON','--execute'],archive_created=False,GPU_calls=0)));return
    assert a.root_review and a.report_qa
    review=archive.read(a.root_review);qa=archive.read(a.report_qa)
    assert review['complete'] is True
    if review['schema']=='R10.root_final_finite_stage_review.v1':
        assert review['method_stage_review_complete'] is True and review['root_actual_joint_views_reviewed']==[86,110,134,179,198]
        assert review['no_pending_R10_GPU_or_experiment_controller'] is True
        ledger=archive.read(d['resources']['path'])
        assert review['R10_actual_nativeF']==ledger['R10']['nativeF'] and review['R10_actual_nativeB']==ledger['R10']['nativeB']
        assert review['R10_actual_smoke_optimizer_updates']==ledger['R10']['smoke_updates']
        assert review['latest_control']==Path(d['final_actual_control_receipt']['path']).parent.name
    else:
        assert review['schema']=='R10.final_method_root_review.v1'
        assert review['all_five_final_views_reviewed'] is True and review['no_more_experiments_or_training'] is True and review['method_snapshot_authorized'] is True
        assert review['resource_ledger_sha256']==d['resources']['sha256']
        assert review['final_actual_control_receipt_sha256']==d['final_actual_control_receipt']['sha256']
    assert type(review['whole_scene_restoration_accepted']) is bool
    if 'report_QA' in review:assert review['report_QA']['sha256']==archive.sha(a.report_qa)
    report_freeze_path=r/'docs/final_report_freeze.actual.json'
    if report_freeze_path.is_file():assert archive.read(report_freeze_path)['final_QA']['sha256']==archive.sha(a.report_qa)
    assert qa.get('decision')=='PASS' or qa.get('passed') is True or qa.get('pass') is True or qa.get('all_checks_passed') is True
    output=a.output or r/'docs/final_method_snapshot_receipt.json';assert not output.exists()
    # Preserve small actual QA/review documents inside the method archive; the HTML stays separate.
    for name,source in [('final_report_QA.actual.json',a.report_qa),('final_root_stage_review.actual.json',a.root_review)]:
        target=r/'docs'/name
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:target.write_bytes(source.read_bytes())
        d['critical_members'].append('docs/'+name)
    d.update(complete=True,method_stage_frozen=True,pending_next_footprint_experiment=False,overall_run_complete=False,
        whole_scene_restoration_accepted=review['whole_scene_restoration_accepted'],root_all_five_final_views_reviewed=True,report_QA_passed=True,
        root_review=archive.pin(a.root_review),report_QA=archive.pin(a.report_qa),GPU_stays_on=True,human_verdict=None,problem_solved=None)
    d['critical_members']=sorted(set(d['critical_members']));archive.write(output,d)
    archive.check_root_signal(r,output)
    print(json.dumps(dict(freeze_signal=archive.pin(output),method_stage_frozen=True,overall_run_complete=False,archive_created=False,GPU_calls=0)))
if __name__=='__main__':main()
