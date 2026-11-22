"""Safety checks only; no Git command or experimental execution."""
from pathlib import Path
import ast,hashlib,json,sys
import publish_road_vehicle_joint as wrapper

OUT=Path(__file__).resolve().parent
def forbidden_git(*args,**kwargs):raise AssertionError('Git invocation forbidden during draft QA')
wrapper.p.git=forbidden_git
pending=json.loads((OUT/'narrative.pending.json').read_text(encoding='utf-8'))
denied=0
try:wrapper.final_narrative(pending)
except RuntimeError:denied+=1
valid=dict(pending,finalized_by_root=True,formal400_and_fresh_reviewed=True,official_DGGT_control_reviewed=True,archive_verified=True,report_QA_passed=True,takeback_verified=True,status_append=wrapper.LABEL,experiment_entry=wrapper.LABEL,failure_append=wrapper.LABEL)
assert wrapper.final_narrative(valid) is valid
for change in [dict(failure_ledger_refs=['V82-F07']),dict(human_verdict=True),dict(formal400_and_fresh_reviewed=False),dict(new_failure_family=True)]:
    try:wrapper.final_narrative(dict(valid,**change))
    except (ValueError,RuntimeError):denied+=1
    else:raise AssertionError('Unsafe final narrative was accepted')
legacy='# 历史结果\n旧数值0.123，原始结果保留。\n'
corrected=wrapper.correction(legacy,'road_source_fidelity_20261122')
assert legacy.split('\n',1)[1] in corrected and 'native_static' in corrected and 'appearance_fit/asset' in corrected
assert wrapper.correction(corrected,'road_source_fidelity_20261122')==corrected
derivation=json.loads((OUT/'isolated_base_derivation.json').read_text(encoding='utf-8'))
assert hashlib.sha256(Path(derivation['upstream']).read_bytes()).hexdigest()==derivation['upstream_sha256']
for name in ['isolated_publication_base.py','publish_road_vehicle_joint.py','prepare_inputs.py','qa_publication_preparation.py']:
    ast.parse((OUT/name).read_text(encoding='utf-8'))
    assert (OUT/name).stat().st_size<300000
assert not (OUT/'selected_files.json').exists() and not (OUT/'narrative.json').exists()
result=dict(passed=True,Git_commands_executed=0,algorithm_runs=0,unsafe_narrative_denials=denied,source_correction_idempotent=True,original_R7_base_unchanged=True,final_configs_created=False,stage_executed=False,commit_created=False,pushed=False)
(OUT/'qa_publication_preparation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result))
