"""CPU-only fail-closed grey asset gate. Standard library only; never imports torch."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import operator
import sys

CHECKS=('component_relations','wheel_layout','visible_silhouette','measurement_fit',
        'visible_geometry_protected','visible_rgb_protected','generated_only_unknown')
NUMERIC_CHECKS={'visible_silhouette','measurement_fit','visible_geometry_protected',
                'visible_rgb_protected','generated_only_unknown'}
OPS={'<=':operator.le,'>=':operator.ge,'==':operator.eq,'<':operator.lt,'>':operator.gt}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def check_gate(path,asset=None,stage='student',schema_test_only=False):
    path=Path(path).resolve(); errors=[]
    try: gate=json.loads(path.read_text())
    except Exception as exc:
        gate={}; errors.append('gate_unreadable:'+type(exc).__name__)
    fixture=gate.get('synthetic_schema_fixture') is True
    if gate.get('schema')!='a20.grey_asset_gate.v2':errors.append('wrong_schema')
    if gate.get('eligible') is not True:errors.append('eligible_not_explicitly_true')
    if gate.get('quality_status')!='grey_approved':errors.append('grey_quality_not_approved')
    if fixture and not schema_test_only:errors.append('synthetic_fixture_forbidden_for_real_entry')
    if schema_test_only and not fixture:errors.append('schema_test_only_requires_explicit_fixture')
    if stage not in gate.get('allowed_stages',[]):errors.append('stage_not_authorized')
    policy=gate.get('threshold_policy',{})
    if policy.get('task_specific') is not True or policy.get('status')!='registered' or not policy.get('source'):
        errors.append('task_specific_threshold_policy_not_registered')

    def pin(item,label):
        if not isinstance(item,dict) or not isinstance(item.get('path'),str) or not item.get('sha256'):
            errors.append(label+':missing_file_binding');return None
        p=Path(item['path'])
        if not p.is_absolute():p=path.parent/p
        p=p.resolve()
        if not p.is_file():errors.append(label+':file_missing');return None
        if sha(p)!=item['sha256']:errors.append(label+':hash_mismatch')
        return p

    artifacts=gate.get('candidate',{}).get('artifacts',[])
    if not artifacts:errors.append('candidate_artifacts_missing')
    pinned=[pin(item,f'candidate_artifacts[{i}]') for i,item in enumerate(artifacts)]
    if asset is not None:
        wanted=Path(asset).resolve()
        if wanted not in pinned:errors.append('requested_asset_not_bound_to_gate')
    entries=gate.get('checks',{})
    for name in CHECKS:
        check=entries.get(name,{})
        if check.get('passed') is not True:errors.append(name+':passed_not_explicitly_true')
        if check.get('status')!='passed':errors.append(name+':status_not_passed')
        criteria=check.get('acceptance_criteria')
        if not isinstance(criteria,list) or not criteria or not all(isinstance(x,str) and x.strip() for x in criteria):
            errors.append(name+':task_acceptance_criteria_missing')
        if not isinstance(check.get('reviewed_by'),str) or not check['reviewed_by'].strip():
            errors.append(name+':review_record_missing')
        evidence=check.get('evidence',[])
        if not evidence:errors.append(name+':evidence_missing')
        for i,item in enumerate(evidence):pin(item,f'{name}.evidence[{i}]')
        rules=check.get('rules',[]); metrics=check.get('metrics',{})
        if name in NUMERIC_CHECKS and (not rules or not check.get('threshold_source')):
            errors.append(name+':task_threshold_rules_missing')
        for i,rule in enumerate(rules):
            key=rule.get('metric');op=rule.get('op');bound=rule.get('bound');value=metrics.get(key)
            good=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
            good=good and isinstance(bound,(int,float)) and not isinstance(bound,bool) and math.isfinite(bound)
            if not good or op not in OPS or not rule.get('unit'):
                errors.append(f'{name}.rule[{i}]:invalid_metric_rule');continue
            if not OPS[op](value,bound):errors.append(f'{name}.rule[{i}]:failed')
    validated=not errors
    # A synthetic schema test may validate, but is never eligible for real GPU entry.
    allowed=validated and not fixture and not schema_test_only
    torch=sys.modules.get('torch')
    cuda_initialized=bool(torch.cuda.is_initialized()) if torch is not None else False
    return dict(schema='a20.grey_asset_gate_check.v2',allowed=allowed,eligible=allowed,
        schema_logic_validated=validated,synthetic_fixture=fixture,schema_test_only=schema_test_only,
        stage=stage,gate=str(path),asset=None if asset is None else str(Path(asset).resolve()),
        reasons=errors,torch_imported=torch is not None,cuda_initialized=cuda_initialized,
        candidate_quality_status=gate.get('quality_status'),user_quality_score_written=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--asset',type=Path)
    ap.add_argument('--stage',choices=['student','material'],default='student')
    ap.add_argument('--schema-test-only',action='store_true')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args(); result=check_gate(args.gate,args.asset,args.stage,args.schema_test_only)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    good=result['schema_logic_validated'] if args.schema_test_only else result['allowed']
    raise SystemExit(0 if good else 2)


if __name__=='__main__':main()
