"""合并本轮实际 native 会话；最新进程快照不得再次叠加。"""
from pathlib import Path
import argparse
import datetime
import json

ROOT_DEFAULT = '/root/autodl-tmp/drivingeditor_repro/campaign/road_source_fidelity_20261122_r2'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(p)


def output_from_argv(argv):
    if '--output' in argv:
        return argv[argv.index('--output') + 1]
    return None


def merge(root):
    root = Path(root).resolve()
    log = root / 'runtime/native_session_log.jsonl'
    expected_path = root / 'recovery/expected_matrix.json'
    expected_stages = read(expected_path).get('expected_stages', []) if expected_path.exists() else []
    events, keys, duplicate_lines = [], set(), []
    for lineno, line in enumerate(log.read_text().splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        identity = (event['PID'], event['started_at'], tuple(event['entry']))
        if identity in keys:
            duplicate_lines.append(lineno)
            continue
        keys.add(identity)
        events.append((lineno, event))
    rows, problems = [], []
    for lineno, e in events:
        stage = output_from_argv(e['entry'])
        if stage is None:
            matches = [v['directory'] for v in expected_stages if v.get('entry_script') == Path(e['entry'][0]).name]
            if len(matches) == 1:
                stage = matches[0]
        if stage is None:
            problems.append(f'Cannot bind session line {lineno} to --output')
        stage_dir = root / stage if stage else root / '_unknown_session'
        contract_path, receipt_path = stage_dir / 'contract.json', stage_dir / 'receipt.json'
        contract = read(contract_path) if contract_path.exists() else {}
        receipt = read(receipt_path) if receipt_path.exists() else {}
        completed = bool(receipt.get('complete', False))
        native_f = int(e['native_context_F'])
        if receipt and int(receipt.get('nativeF', native_f)) != native_f:
            problems.append(f'{stage}: receipt nativeF does not equal closed context')
        # Both current executable contracts forbid gradient optimizer updates;
        # closed contexts from failed runs remain charged to the cumulative F.
        recognized = Path(e['entry'][0]).name in ('probe_source_chain.py', 'solve_fixed_source_rgb.py', 'probe_static_original_source.py')
        zero_update_contract = contract.get('optimizer_updates') == 0 or contract.get('optimizer_gradient_updates') == 0
        native_b = receipt.get('nativeB', 0 if recognized and zero_update_contract else None)
        if native_b is None:
            problems.append(f'{stage}: nativeB has no explicit receipt or recognized zero-update contract')
        gradient_updates = receipt.get('gradient_optimizer_updates', receipt.get('optimizer_updates', 0 if zero_update_contract else None))
        if gradient_updates is None:
            problems.append(f'{stage}: optimizer update accounting is unspecified')
        if e.get('new_diffusion_calls', 0) != 0 or receipt.get('new_diffusion_calls', 0) != 0:
            problems.append(f'{stage}: nonzero diffusion outside this task contract')
        if receipt.get('target_migration', False):
            problems.append(f'{stage}: target migration outside this task contract')
        rows.append(dict(stage=stage, PID=e['PID'], session_log_line=lineno,
            command=e['entry'], nativeF=native_f, nativeB=native_b,
            gradient_optimizer_updates=gradient_updates, RGB_linear_solves=receipt.get('RGB_linear_solves', 0),
            gradient_measurements=contract.get('gradient_measurements', 0),
            new_diffusion_calls=e.get('new_diffusion_calls', 0),
            completed_receipt=completed, failed_or_incomplete_receipt=not completed,
            started_at=e['started_at'], ended_at=e['ended_at'],
            wall_seconds=e['ended_at'] - e['started_at'],
            contract=str(contract_path), receipt=str(receipt_path),
            original_asset_modified=receipt.get('original_asset_modified'),
            original_event=e))
    snapshot_path = root / 'resource_ledger.json'
    snapshot = read(snapshot_path) if snapshot_path.exists() else None
    lease = read(root / 'gpu_lease.json') if (root / 'gpu_lease.json').exists() else {}
    if lease.get('active', False):
        problems.append('Native GPU lease is still active; latest process may not have closed its session')
    totals = dict(nativeF=sum(r['nativeF'] for r in rows),
                  nativeB=sum(r['nativeB'] or 0 for r in rows),
                  gradient_optimizer_updates=sum(r['gradient_optimizer_updates'] or 0 for r in rows),
                  RGB_linear_solves=sum(r['RGB_linear_solves'] for r in rows),
                  gradient_measurements=sum(r['gradient_measurements'] for r in rows),
                  completed_native_contexts=len(rows),
                  native_context_wall_seconds=sum(r['wall_seconds'] for r in rows),
                  new_diffusion_calls=sum(r['new_diffusion_calls'] for r in rows),
                  target_training_updates=0, target_migration_updates=0, power_operations=0)
    cpu_path = root / 'color_chain_audit/aggregate_cpu_audit.json'
    cpu_resources = None
    if cpu_path.exists():
        cpu = read(cpu_path)
        extra = int(cpu.get('parent_source_RGB_solves_are_separate_from_CPU_precision_replay_channel_solves', 0))
        totals['extra_CPU_precision_replay_channel_solves'] = extra
        totals['combined_RGB_linear_numerical_solves'] = totals['RGB_linear_solves'] + extra
        cpu_resources = dict(evidence=str(cpu_path), all_checks_passed=cpu.get('all_checks_passed'),
            extra_precision_replay_channel_solves=extra, source_fit_channel_solves=totals['RGB_linear_solves'],
            model_and_linear_state_assets_modified=cpu.get('model_and_linear_state_assets_modified'),
            precision_replay_is_not_an_additional_source_fit=True)
        if cpu.get('parent_actual_native_totals', {}).get('nativeF') != totals['nativeF']:
            problems.append('Independent CPU nativeF accounting differs from closed session log')
    return dict(schema='source_fidelity_cumulative_resource.v1', complete=not problems,
                run_id=root.name, source_log=str(log), sessions=rows, totals=totals,
                duplicate_session_lines_not_counted_twice=duplicate_lines,
                latest_process_snapshot=snapshot,
                latest_snapshot_policy='Corroboration only; never added to cumulative session counts',
                gpu_lease_snapshot=lease, accounting_problems=problems,
                independent_CPU_audit_resources=cpu_resources,
                nativeB_known_for_every_closed_session=all(r['nativeB'] is not None for r in rows),
                source_linear_solves_are_not_gradient_updates=True,
                scope='本轮实际结束会话，包含失败/未采用现场；不叠加旧轮次资源',
                snapshot_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=ROOT_DEFAULT)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    result = merge(args.root)
    output = args.output or str(Path(args.root) / 'recovery/resource_ledger_cumulative.json')
    write(output, result)
    print(json.dumps(dict(complete=result['complete'], totals=result['totals'], problems=result['accounting_problems'], output=output)))


if __name__ == '__main__':
    main()
