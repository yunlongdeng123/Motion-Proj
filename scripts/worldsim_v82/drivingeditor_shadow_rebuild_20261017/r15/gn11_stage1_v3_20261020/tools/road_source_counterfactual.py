"""登记同 seed / 预算的去道路来源配对计划，或比较实际终态；不授权 GPU。"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np


def plan_pair(plan_path, output, query_raw=89, actor=37, source_raw=None):
    plan = json.loads(plan_path.read_text())
    bundle = Path(plan['input_bundle'])
    metadata = json.loads(bundle.read_text())
    with np.load(bundle.parent/metadata['units'], allow_pickle=False) as data:
        ids = np.unique(data['query_unit_id'][(data['query_frame'] == query_raw-89) & (data['query_actor'] == actor)])
        ids = ids[data['provenance'][ids] == 1]
        scores = data['base_weight'][:, ids].sum(1)
        if source_raw is None:
            source_raw = 89+int(scores.argmax())
        if not 89 <= source_raw <= 137 or scores[source_raw-89] <= 0:
            raise ValueError('chosen source has no hard-eligible query observation; not a meaningful ablation')
        eligible = int((data['base_weight'][source_raw-89, ids] > 0).sum())
    if output.exists():
        raise FileExistsError('never overwrite an experiment plan or result')
    output.mkdir(parents=True)
    for kind in ['reference', 'drop_source']:
        candidate = copy.deepcopy(plan)
        candidate['counterfactual_id'] = f'raw{query_raw}_a{actor}_drop_raw{source_raw}_{kind}'
        candidate['road_source_exclusions'] = [] if kind == 'reference' else [dict(source_raw=source_raw, query_raw=query_raw, actor=actor)]
        (output/f'{kind}_plan.json').write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    manifest = dict(task='WS-V82-R15-GN11-STAGE1-V3-20261020',
                    run_id=f'road_cf_raw{query_raw}_a{actor}_drop_raw{source_raw}',
                    status='PLANNED_ONLY_SEPARATE_GPU_AUTHORIZATION_REQUIRED', query_raw=query_raw, actor=actor,
                    source_raw=source_raw, hard_eligible_query_units=eligible,
                    change='only direct road-source observations of these world query units are removed',
                    native_training_frame_removed=False, native49_reconstruction_unchanged=True,
                    seed=0, fresh_updates_each=30000, second_stage_updates=0,
                    additional_updates_if_both_executed=60000, updates_executed=0,
                    possible_reuse='exact matching full-arm fresh30K reference may be reused after contract audit',
                    optimizer_and_rng_state_must_match_at_start=True,
                    required_outputs=['paired full49 joint/static/road render', 'fixed-world-unit numeric differences', 'held-source residual diagnostic, not independent test GT'],
                    GPU_authorized=False, help_proven=False, human_verdict=None,
                    failure_ledger_refs=['V82-F01', 'V82-F06'], failure_ledger_delta='none')
    (output/'counterfactual_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def compare(reference, candidate, output):
    before, after = [json.loads((path/'metadata.json').read_text()) for path in [reference, candidate]]
    if any(value['optimizer_updates'] != 30000 for value in [before, after]):
        raise ValueError('both matched full30K states are required; CPU weights are not causal evidence')
    if before['version'] != after['version'] or before['input_bundle'] != after['input_bundle']:
        raise ValueError('different road representation or input')
    if not before.get('paired_training_config') or before['paired_training_config'] != after.get('paired_training_config'):
        raise ValueError('native initialization/seed/loss/sampler/schedule configs are not paired')
    if before['source_exclusions'] or len(after['source_exclusions']) != 1:
        raise ValueError('expected unmodified reference and one explicit source exclusion')
    with np.load(reference/'optimization.npz', allow_pickle=False) as a, np.load(candidate/'optimization.npz', allow_pickle=False) as b:
        if not np.array_equal(a['unit_ids'], b['unit_ids']):
            raise ValueError('world unit identities differ')
        bundle = Path(before['input_bundle'])
        bundle_meta = json.loads(bundle.read_text())
        exclusion = after['source_exclusions'][0]
        with np.load(bundle.parent/bundle_meta['units'], allow_pickle=False) as units:
            query_ids = np.unique(units['query_unit_id'][(units['query_frame'] == exclusion['query_raw']-89)
                                                        & (units['query_actor'] == exclusion['actor'])])
        query_mask = np.isin(a['unit_ids'], query_ids)
        if not query_mask.any():
            raise ValueError('paired assets have no supported world query units')
        result = dict(status='PAIRED_ASSET_DIFFERENCES_NOT_AUTOMATIC_QUALITY_IMPROVEMENT',
                      color_L1_mean=float(np.abs(a['color']-b['color']).mean()),
                      model_position_L2_mean=float(np.linalg.norm(a['model_xyz']-b['model_xyz'], axis=1).mean()),
                      exclusions=after['source_exclusions'], human_verdict=None,
                      supported_query_units=int(query_mask.sum()),
                      query_color_L1_mean=float(np.abs(a['color'][query_mask]-b['color'][query_mask]).mean()),
                      query_position_L2_m_mean=float(np.linalg.norm(a['model_xyz'][query_mask]-b['model_xyz'][query_mask], axis=1).mean()/bundle_meta['meter_to_model']),
                      full49_render_and_regional_quality_review_still_required=True)
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        np.savez_compressed(output/'per_unit_differences.npz', unit_ids=a['unit_ids'],
                            is_requested_query=query_mask, color_delta=b['color']-a['color'], model_position_delta=b['model_xyz']-a['model_xyz'])
    (output/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def execute_pair(root):
    # 独立审批范围，主四臂审批不能静默扩大为另外60K。
    approval_path = root/'GPU_APPROVAL.json'
    if not approval_path.is_file():
        raise RuntimeError('separate counterfactual GPU_APPROVAL.json is absent')
    approval = json.loads(approval_path.read_text())
    manifest = json.loads((root/'counterfactual_manifest.json').read_text())
    plan = json.loads((root/'reference_plan.json').read_text())
    if (approval.get('status') != 'USER_AUTHORIZED_ROAD_SOURCE_COUNTERFACTUAL'
            or approval.get('scope') != 'two_zero_update_gates_and_two_fresh30K'
            or approval.get('road_version') != plan['road_version']
            or approval.get('source_raw') != manifest['source_raw']
            or approval.get('query_raw') != manifest['query_raw']
            or approval.get('actor') != manifest['actor']):
        raise RuntimeError('counterfactual authorization does not match the exact pair')
    if (root/'execution_status.json').exists():
        raise FileExistsError('prior execution exists; preserve failure and inspect, never auto-restart')
    import fcntl
    lock = (root/'counterfactual_controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(root).free < plan['minimum_campaign_free_bytes']:
        raise RuntimeError('insufficient disk admission for paired complete states')
    status = dict(status='running', completed=[], automatic_restart=False, extra_updates_planned=60000)
    def save():
        temporary = root/'execution_status.partial.json'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(root/'execution_status.json')
    save()
    try:
        for kind in ['reference', 'drop_source']:
            output = root/kind
            output.mkdir()
            for mode, destination in [('gate', output/'gate_001'), ('train', output/'run30k')]:
                command = [plan['python'], str(Path(plan['campaign'])/'run_v3_arm.py'),
                           '--engine', str(Path(plan['campaign'])/'engine'), '--output', str(destination),
                           '--bundle', plan['input_bundle'], '--baseline-config', plan['baseline_config'],
                           '--readiness', plan['readiness'], '--binding-tools', plan['binding_tools'],
                           '--gpu-lock', plan['gpu_lock'], '--plan', str(root/f'{kind}_plan.json'),
                           '--arm', 'full', '--mode', mode]
                if mode == 'train':
                    command.extend(['--gpu-gate', str(output/'gate_001/gpu_gate_receipt.json')])
                subprocess.run(command, check=True)
            status['completed'].append(kind)
            save()
        status['status'] = 'complete_full49_review_pending'
        save()
    except BaseException as error:
        status.update(status='failed_preserved_no_restart', error=repr(error))
        save()
        raise
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--query-raw', type=int, default=89)
    parser.add_argument('--actor', type=int, default=37)
    parser.add_argument('--source-raw', type=int)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-after-user-gpu-approval', action='store_true')
    args = parser.parse_args()
    result = (execute_pair(args.output) if args.execute_after_user_gpu_approval else
              plan_pair(args.plan, args.output, args.query_raw, args.actor, args.source_raw) if args.plan else
              compare(args.reference, args.candidate, args.output))
    print(json.dumps(result, indent=2))
