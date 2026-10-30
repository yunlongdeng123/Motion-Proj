"""重建 V43 设计产物；仅更新首页设计块，不重跑或改写场景诊断与实验结果。"""
from pathlib import Path
import json
import re

from build_architecture import architecture

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'outputs/V43_Ownership_20261027'
START = '<!-- v43-design-revision3-start -->'
END = '<!-- v43-design-revision3-end -->'

from revision4_contract import DESIGN, DESIGN_HTML, upgrade_schema, upgrade_state, experiment_plan


def schema():
    return {
        'schema_version': 'v43-ownership-design-3',
        'implementation_status': 'design_sidecar_not_full_checkpoint_migration',
        'gaussian_key': ['scene_id', 'checkpoint_sha256', 'branch', 'row_id', 'lineage_id'],
        'semantic': {'class_id': None, 'status': 'unknown', 'candidates': []},
        'ownership': {'kind': None, 'instance_or_surface_id': None, 'status': 'unknown', 'candidates': [], 'stable_across_frames': True},
        'motion': {'model': None, 'frame_id': None, 'valid_time_interval': None, 'status': 'unknown'},
        'evidence': {
            'status': 'unreviewed', 'records': [], 'confidence': None, 'calibration_version': None,
            'unknown_reason': [],
            'record_fields': ['source_kind', 'source_frame', 'source_row_or_pixel', 'visibility', 'support_weight', 'correlation_group', 'version', 'reviewer'],
            'unknown_reason_vocabulary': ['missing_observation', 'ambiguous_instance', 'conflicting_observations', 'mixed_spatial_support', 'unverified_geometry', 'unverified_effect'],
        },
        'support_type': {'value': 'unknown', 'vocabulary': ['pure', 'spatial_mixed', 'unknown'], 'evidence_refs': []},
        'effect': {
            'type': 'unknown', 'vocabulary': ['none', 'shadow', 'unknown'],
            'effect_owner_id': None, 'receiver_surface_id': None,
            'receiver_frame': None, 'confidence': None, 'evidence_refs': [],
            'shadow_contract': 'car-owned appearance effect on a world-bound road receiver; road geometry is retained',
        },
        'time_state': {'timestamp': None, 'visibility': 'unknown', 'occlusion': 'unknown', 'observation_confidence': None, 'evidence_refs': []},
        'lineage': {
            'parent_keys': [], 'child_keys': [], 'operation': None, 'operation_version': None,
            'two_sided_support_refs': [], 'initialization_record': None, 'local_refit_run_id': None,
            'original_parameters_preserved': True, 'before_after_contribution_refs': [],
        },
        'legacy_render_assignment': {'branch': None, 'actor_id': None, 'verified_complete': False},
        'target_query': {
            'one': 'confirmed ownership matches requested target',
            'zero': 'confirmed ownership excludes requested target; may be another actor',
            'unknown': 'ownership not established or mixed/conflicting',
        },
        'routing': {
            'pure': 'assign owner if evidence is sufficient; correct geometry separately when needed',
            'projected_dual_domain': 'check parent ellipsoid and observed support compatibility; do not treat projected overlap as spatial separability',
            'spatial_mixed': 'only split and locally refit when spatial support is compatible; retain both children and original parent',
            'spatially_incompatible': 'geometry correction or surface remodeling; preserve unknown rather than force a centroid-midplane split',
            'shadow': 'decouple effect owner from world receiver geometry',
            'insufficient_or_conflicting': 'abstain as unknown',
        },
        'shadow4': [
            {'body': True, 'shadow': True, 'role': 'original reconstruction'},
            {'body': False, 'shadow': True, 'role': 'body off, shadow retained'},
            {'body': True, 'shadow': False, 'role': 'shadow off, car and unshadowed road retained'},
            {'body': False, 'shadow': False, 'role': 'target deletion'},
        ],
        'invariants': [
            'semantic unknown does not erase independently verified instance identity',
            'stationary or occluded actor retains stable owner; visibility and confidence are separate',
            'render batch static does not imply background semantic class',
            'legacy renderer owner is prior operational assignment, not verified provenance',
            'unknown is field-level abstention, not a new background class',
            'spatial support and appearance effect may coexist and are not interchangeable',
            'both sides of a mixed Gaussian are retained and independently checked after refit',
            'no-edit radiometric fidelity does not establish two-side separation under after-edit transmittance',
            'failed or unverified after-edit separation retains unknown and does not establish editable ownership',
            'moment matching does not imply exact pixelwise alpha or RGB conservation',
            'original checkpoint and parent Gaussian parameters are preserved',
            'GT boxes remain proposal/read-only audit, never point or mask filtering labels',
            'after-render diagnostics and generated images do not recursively become GT',
            'R2 ROI whitening, unknown SH whitening, classifier learning and editing benefit are separate',
            'scene 1 is classifier inference/development only, not whole-method unseen testing',
        ],
        'human_verdict': None,
    }


def update_index(page):
    page = re.sub(r'<title>.*?</title>', '<title>V43 · 局部可编辑联合分解</title>', page, count=1, flags=re.S)
    page = re.sub(r'<h1>.*?</h1>', '<h1>V43：面向对象编辑的局部可编辑联合分解</h1>', page, count=1, flags=re.S)
    page = re.sub(r'(<h1>.*?</h1>)<p>.*?</p>',
        r'\1<p>Revision 4：共享证据、联合输出、分阶段训练。场景判断与编辑指令分离；unknown优先召回，实际修改须有依据。下一步是小型机制数据与理想上界，环境纳入固定分层验收。本次只修订设计，没有新增训练或渲染。</p>', page, count=1, flags=re.S)
    start4, end4 = '<!-- v43-design-revision4-start -->', '<!-- v43-design-revision4-end -->'
    block = start4 + '\n' + DESIGN_HTML + '\n' + end4
    if start4 in page:
        page = re.sub(re.escape(start4) + r'.*?' + re.escape(end4), lambda _: block, page, count=1, flags=re.S)
    elif START in page:
        page = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda _: block, page, count=1, flags=re.S)
    else:
        raise ValueError('缺少已知设计区块，停止以保护历史内容')
    page = re.sub(r'<p><a href="ownership_record_contract.json">四字段与旧身份兼容合同</a>.*?</p>',
        '<p><a href="ownership_record_contract.json">联合状态与旧身份兼容合同</a>：共享模型联合判断与修复，分阶段训练；小型机制数据与理想上界先行。旧规则表为历史数据，不是新模型的输出。</p>', page, count=1, flags=re.S)
    page = re.sub(r'<section><h2>时间状态与后续实验</h2>.*?</section>',
        '<section><h2>时间状态与后续实验</h2><p>稳定身份与逐帧可见性分开；允许多帧联合推断，不将自身预测循环当GT。机制上界与环境四路配对可并行，随后做候选范围和小规模联合学习。场景1仅推理与开发诊断。</p><p>V42显露背景修复和按需生成仍保留；<a href="../Joint_Decomposition_Repair_20261027/index.html">旧联合修复对照</a>作为历史方案保存，联合收益必须由匹配覆盖与预算的实验建立。</p></section>', page, count=1, flags=re.S)
    page = re.sub(r'<footer>.*?</footer>', '<footer>设计revision4 / 新路线实验待执行 / 旧R2与204帧白材质保持 / 本对话禁止自动AutoDL关机 / 人工verdict未填</footer>', page, count=1, flags=re.S)
    return page


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = OUT / 'index.html'
    if not index_path.exists():
        raise FileNotFoundError('保留历史首页是本生成器的前提，缺失时不新建替代页')
    page = update_index(index_path.read_text(encoding='utf-8-sig'))
    (OUT / 'V43_DESIGN.md').write_text(DESIGN, encoding='utf8')
    (OUT / 'architecture.svg').write_text(architecture(), encoding='utf8')
    (OUT / 'ownership_record_contract.json').write_text(json.dumps(upgrade_schema(schema()), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    state = {
        'architecture_revision': 3,
        'priority': 'identity_and_representation_repair_then_background_and_generation',
        'scope': 'local_design_revision_only',
        'new_training_updates_this_revision': 0, 'remote_compute_this_revision': False,
        'training_update_count_scope': 'document_generation_only; actual short-probe child updates are listed separately',
        'existing_gpu_diagnostics_and_structural_probes_preserved': True,
        'existing_unknown_white_group_frames_complete': 204, 'unknown_white_rerun_this_revision': False,
        'v1_native_no_edit_path_preserved': True, 'v42_full_edit_repair_generation_architecture_preserved': True,
        'manual_GT_complete': False, 'manual_review_complete': False, 'human_verdict': None,
        'oracle_ownership_control_complete': False, 'learnable_multifield_classifier_implemented': False,
        'two_sided_local_split_refit_probe_implemented': True, 'unified_spatially_valid_split_chain_complete': False,
        'current_short_probe': {'raws': [98, 113, 114], 'projected_dual_domain_candidates': [452, 777, 810],
            'both_centroids_within_3sigma': [0, 0, 0], 'raw98_top8_tiny_tail_count': 8,
            'refit_raw': 113, 'refit_parents': 7, 'refit_children': 14, 'child_radiometric_parameter_updates': 200,
            'checkpoint200_restore_additional_updates': 0, 'classifier_training_updates': 0,
            'spatial_separation_accepted': False, 'editing_benefit_proven': False,
            'evidence': 'gpu_v43/editable_ownership_20261030/index.html'},
        'shadow4_implemented': False,
        'checkpoint_schema_migrated': False, 'joint_superiority_proven': False,
        'scene1_classifier_role': 'inference_and_development_diagnostics_only_no_classifier_training',
        'first_experiment_order': ['ideal_or_evidence_candidate_ceiling_and_mixed_detection', 'two_sided_split_refit_and_shadow4', 'external_classifier_training_and_transfer'],
    }
    state = upgrade_state(state)
    (OUT / 'mechanism_experiment_plan.json').write_text(json.dumps(experiment_plan(), ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (OUT / 'direction_state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    index_path.write_text(page, encoding='utf8')
    print(json.dumps({'report': str(index_path), 'architecture_revision': 4, 'scope': 'design_only_no_diagnostic_or_experiment_rebuild'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
