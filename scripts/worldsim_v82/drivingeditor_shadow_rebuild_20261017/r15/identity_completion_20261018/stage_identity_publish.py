"""Copy only this task's evidence/source; preserve unrelated checkout dirt."""
from pathlib import Path
import shutil,json,os

W=Path(__file__).resolve().parent
P=W.parent/'r15_closure_run/publish_repo'
REL='docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/identity_20261018'
SREL='scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/identity_completion_20261018'
OUT=W.parents[1]/'outputs/identity_comparison_20261018'

def long(p):return Path('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p
def cp(src,dst):
    src,dst=long(src),long(dst);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)

def main():
    for p in (W/'bundle').glob('*.py'):
        if p.name!='inspect_tristate.py':cp(p,P/SREL/'code'/p.name)
    for name in ['cpu_checks.py','build_tristate_report.py','build_comparison_report.py']:
        cp(W/name,P/SREL/name)
    for p in (W/'metadata/infra').glob('*.py'):cp(p,P/SREL/'original_infra_snapshot'/p.name)
    for name in ['spec5.json','old_effective_config.json','preserved_inputs.json','prepared.json',
                 'mask_coverage.json','mask_coverage_R15_partial.json','execution_policy.json']:
        cp(W/'metadata'/name,P/REL/'inputs'/name)
    cp(W/'metadata/native_masks_for_review/manifest.json',P/REL/'inputs/native_mask_manifest.json')
    cp(W/'cpu_checks.json',P/REL/'cpu_checks.json')
    # Keep binary full native mask proposals with the prepared input evidence.
    for p in (W/'metadata/native_masks_for_review').glob('*.npz'):cp(p,P/REL/'native_mask_inputs'/p.name)
    for model in ['r15','old3','old5']:
        receipt=OUT/'assets'/model/'render_receipt.json'
        if not receipt.exists():receipt=W/'metadata'/('renders_'+model)/'render_receipt.json'
        if receipt.exists():cp(receipt,P/REL/(model+'_render_receipt.json'))
    for p in (OUT/'tristate').rglob('*'):
        if p.is_file():cp(p,P/REL/'tristate'/p.relative_to(OUT/'tristate'))
    complete=json.loads((OUT/'completion_status.json').read_text()) if (OUT/'completion_status.json').exists() else {}
    projection=json.loads((OUT/'tristate/build_status.json').read_text())
    status=dict(task_id='WS-V82-R15-IDENTITY-COMPARISON-20261018',
        R15_frozen_4k_identity_renders_complete=True,R15_render_cases=245,old3_render_cases=147,
        old5_30k_training='complete30000 and real-ID rendering complete' if complete.get('old5_30k_complete') else 'running, maximum 30000 updates; not a completed baseline result',
        CPU_tristate_audit='complete',GPU_tristate_projection='complete' if projection['GPU_projection_complete'] else 'queued after old5 training and rendering',
        original_R15_checkpoint_modified=False,frozen_R15_render_parameter_updates=0,no_GN11_updates=True,no_subagents=True,
        repaired_R15_4k='separate fresh run; real GPU smoke passed, training in progress; not included in frozen comparison',
        failure_ledger_refs=['V82-F06'],failure_ledger_delta='none',human_verdict=None,problem_solved=None,
        remote_task='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/identity_completion_20261018',
        checkpoint_preservation='GN11 original30K, historical instance3 30K, R15 paused4K remain untouched',
        shutdown='user authorized after all GPU/CPU/controller work, saved delivery and push; not yet executed')
    target=long(P/REL/'status.json');target.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'progress.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(source=SREL,evidence=REL)))

if __name__=='__main__':main()
