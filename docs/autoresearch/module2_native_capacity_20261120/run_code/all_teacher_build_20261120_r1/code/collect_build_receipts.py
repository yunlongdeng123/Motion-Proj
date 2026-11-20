"""CPU-only closure metadata for the completed fixed BUILD queue. No GPU imports."""
from pathlib import Path
import json
from build_inputs import ROOT, B, FRAMES, read, write, sha


def collect():
    result = read(ROOT / 'results.json')
    if result.get('completed') is not True:
        raise RuntimeError('registered BUILD queue is not complete')
    field, numeric, eng = result['field'], result['numerical'], result['engineering']
    binding = read(ROOT / 'source_freeze.json')['binding']
    changed = [p for p, digest in binding.items() if sha(p) != digest]
    if changed:
        raise RuntimeError('registered immutable source changed: ' + repr(changed))
    field_physical, number_physical = field['physical'], numeric['physical']
    es, en, initial = eng['field']['physical'], eng['numeric']['physical'], eng['initial_readonly']
    driver_f = (es['native_forward_calls'] + field_physical['native_forward_calls'] +
                field['endpoint_readonly_nativeF'] + en['nativeF'] + number_physical['nativeF'])
    if driver_f != result['actual_native_forward_calls']:
        raise RuntimeError('actual forward counter does not match phase accounting')
    if field['logical_steps'] != 1200 or number_physical['FG'] > 2000:
        raise RuntimeError('formal BUILD budget mismatch')
    account = dict(
        task_id=result['task_id'], completed=True, data_role='BUILD_fit_all_qualified',
        formal=dict(shared_Adam_updates=field_physical['parameter_updates'],
                    numeric_FG=number_physical['FG'], numeric_accepted_steps=number_physical['accepted']),
        engineering=dict(shared_Adam_updates=es['parameter_updates'], numeric_FG=en['FG'],
                         numeric_accepted_steps=en['accepted'], initialization_optimizer_updates=0),
        actual_native_forward_calls=driver_f + initial['actual_nativeF'],
        actual_native_per_view_backward_calls=es['native_backward_calls'] + field_physical['native_backward_calls'] +
                                              en['nativeB'] + number_physical['nativeB'],
        actual_torch_backward_invocations=es['backward_calls'] + field_physical['backward_calls'] +
                                          en['torchBackward'] + number_physical['torchBackward'],
        failed_shared_step_attempts=es['failed_step_attempts'] + field_physical['failed_step_attempts'],
        elapsed_initialization_seconds=initial['seconds'], elapsed_queue_seconds=result['seconds'],
        nonoverlap_elapsed_seconds=initial['seconds'] + result['seconds'],
        public1920_forward_calls='independent public phase; excluded from this accounting',
        numerical_FG_note='Two same-prediction scalar-precision evaluations belong only to engineering; accepted Fortran steps are not torch.optimizer.step calls',
        end_semantics='Training completion is execution completion, not image quality acceptance or strict representation bound',
        GPU_keep_on=True)
    write(ROOT / 'execution_accounting.json', account)
    own = [ROOT / 'protocol.json', ROOT / 'README.md', ROOT / 'CPU_inputs.json', ROOT / 'CPU_tests_stdout.txt']
    own += sorted((ROOT / 'code').glob('*.py'))
    own_files = {str(p): sha(p) for p in own}
    outputs = {}
    for role, folder in [('common', ROOT / 'common'), ('field', ROOT / 'field/formal_r1'),
                         ('numerical', ROOT / 'numerical/formal_r1')]:
        files = ['receipt.json']
        files += ['initial_ci.npz', 'initial_field.pt', 'common_loss_snapshot.json'] if role == 'common' else ['oracle_colors.pt', 'asset/road.npz', 'asset/provenance.npz', 'asset/persistent/scene.json', 'asset/persistent/proof.json']
        if role == 'field':
            files.append('field.pt')
        outputs[role] = {str(folder / name): sha(folder / name) for name in files}
    reused = {
        'shared_state': str(B / 'shared_field/code/shared_training_state.py'),
        'ordinary_GS_world_read': str(B / 'shared_field/code/shared_field_gs_read.py'),
        'native_objective': str(B / 'code/native_capacity_loss.py'),
        'private_Fortran_complete_workspace': str(B / 'numerical_completion/code/checkpointed_lbfgsb.py'),
        'reduction_only_precision': str(B / 'numerical_completion/code/fp64_reduction_loss.py'),
        'original_complete_crossprocess_lifecycle': str(B / 'numerical_completion/smoke_r1/receipt.json')}
    source_inventory = dict(completed=True, scientific_source_frozen=True, new_namespace_only=True,
        own_files=own_files, immutable_binding_count=len(binding), reused_files={p: sha(p) for p in reused.values()},
        reused_roles=reused, output_authorities=outputs,
        native_environment='/root/autodl-tmp/envs/drivingeditor/bin/python (Python3.9, SciPy1.13.1 ABI)',
        CPU_tests=dict(module='test_build_inputs', count=5, stdout=str(ROOT / 'CPU_tests_stdout.txt')))
    write(ROOT / 'owned_source_inventory.json', source_inventory)
    closure = dict(completed=True, source_binding_all_unchanged=True, source_binding_files=len(binding),
        initial_ci_exact_original=initial['ci0_exact_original'], shared_field_actual_query_exact=field['field_reload_query_max_abs'] == 0,
        shared_derived_Boundedci_exact=field['derived_Boundedci_equal'],
        numeric_export_point=numeric['extra']['export_point'], numeric_training_converged=numeric['numeric_training_converged'],
        strict_numerical_negative_certificate=False, no_validation_claim=True,
        public1920_pending=True, default_scene_modified=False, quality_PASS=False,
        execution_accounting=str(ROOT / 'execution_accounting.json'), inventory=str(ROOT / 'owned_source_inventory.json'))
    write(ROOT / 'closure_receipt.json', closure)
    print(json.dumps(dict(accounting=account, closure=closure)), flush=True)


if __name__ == '__main__':
    collect()
