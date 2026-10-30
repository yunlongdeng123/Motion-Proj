"""Revision 5：独立世界GT、真实重建污染与合法输入隔离；本模块不调度实验。"""
from pathlib import Path

from revision4_contract import upgrade_schema as upgrade_schema_r4
from revision4_contract import upgrade_state as upgrade_state_r4


DESIGN = Path(__file__).with_name('design_revision5.md').read_text(encoding='utf8')
DESIGN_HTML = '''<section id="v43-design-scope"><h2>Revision 5：世界表面真值与真实重建污染</h2>
<p>GT定义在独立仿真世界的表面、对象与编辑干预结果，不预设每颗Gaussian有唯一标签。保留共享证据、多头联合输出、分阶段训练，以及V1原生直出和V42大框／内部组件。</p>
<p><strong>该文档生成器未运行训练或场景渲染。</strong>先完成1个最小普通场景的实际重建，检查自然污染，再扩3–5机制场景、小规模学习或批量造数；不承诺重建必然出现指定错误。</p>
<p><a href="V43_DESIGN.md">完整设计与实验合同</a> · <a href="architecture.svg">主架构SVG</a> · <a href="architecture.png">主架构PNG</a> · <a href="data_architecture.svg">数据与GT隔离SVG</a> · <a href="data_architecture.png">数据与GT隔离PNG</a> · <a href="ownership_record_contract.json">表示合同</a> · <a href="mechanism_experiment_plan.json">待执行矩阵</a></p></section>
<section id="v43-ground-truth"><h2>独立世界GT；Gaussian参考保留歧义</h2>
<table><tr><th>量／状态</th><th>定义与边界</th></tr>
<tr><td>世界／对象／干预GT</td><td>独立表面、对象、深度、可见性，以及对象删除／横移／转向后的重新渲染结果</td></tr>
<tr><td>原分支、网页语义</td><td>带噪证据；static／dynamic和颜色均不能当正确归属</td></tr>
<tr><td>alpha×T × 像素语义</td><td>指定相机／时刻／可见状态及对应有效域内的派生责任，不是Gaussian GT或身份概率</td></tr>
<tr><td>错深度／mixed／无对应／缺测</td><td>分别记录几何冲突、多支持、no_correspondence、missing_observation，不按二维重叠强行硬标签</td></tr></table>
<p>对应须核对独立表面、深度和可见性；投影跨类不足以证明三维mixed。模型自身预测、删点图、after暗像素和生成图不能循环成为GT。真实三场景没有完整GT，人工／subagent只提供带范围和分歧的证据参考；人工verdict保持空。</p></section>
<section id="v43-joint-model"><h2>共享证据 → 联合场景判断 → 编辑决策</h2>
<table><tr><th>联合输出</th><th>用途与边界</th></tr>
<tr><td>语义／实例或表面／运动／证据</td><td>稳定身份与当前可见性分开；场景判断不接收delete／shift／rotate指令</td></tr>
<tr><td>派生混合责任、几何可靠性与修正</td><td>改归属、修几何、保留两侧拆分、有限新增或暂缓共同建模</td></tr>
<tr><td>车辆关联效应</td><td>world道路receiver与vehicle effect owner分开；道路几何不随车移动</td></tr>
<tr><td>不确定性／补证据</td><td>认知后验、混合责任、当前可见性独立；未校准score不称概率</td></tr></table>
<p><strong>unknown候选不等于操作集合。</strong>既有允许域内优先召回，正常背景候选只记录误召回／成本；真正操作再核验保留域。相同观察更换请求时身份保持，动作可以不同。训练参考只在训练分区作为监督，不混入部署特征；验证／测试GT只进隔离诊断和评价。</p>
<p>四个强消融保持：<strong>特征阈值／仅概率决策／仅表示修复／两者联合</strong>；B3内比较顺序冻结、联合梯度、跨头stop-gradient。匹配候选和编辑覆盖，暂缓目标不移出残留分母。</p></section>
<section id="v43-data-route"><h2>主路线：普通世界 → 受限观测 → 当前代码真实重建</h2>
<img src="data_architecture.svg" alt="V43普通世界、受限RGB稀疏LiDAR、实际重建、V43及隔离GT评价组件图">
<p>现成／程序化普通车、道路和灯光，沿受限轨迹生成RGB与稀疏LiDAR，输入当前重建代码真实训练；保留初始化、活跃损失、预算、seed、checkpoint与接口适配。实际产生的污染进入V43，独立表面／深度／编辑参考负责核验。完整网格／稠密深度／测试编辑答案不进入部署输入。</p>
<p>先1个最小场景检查接口和实际污染，再3–5机制集，随后小学习／批量。未出现指定错误时如实报告，不改写为覆盖成功。人工标签遮挡、错归属／错深度或可追溯混合仅作辅助控制，不能替代自然污染证据。</p>
<p>配对编辑由独立世界干预后重渲染，不由待测高斯删点生成。阴影为<strong>world road receiver + vehicle effect owner</strong>；shadow4需独立控制车体可见与车辆投影。<strong>有车减无车只能先称对象移除总影响，不能直接当纯阴影GT</strong>，须排除遮挡、反射、间接光等混入。</p></section>
<section id="v43-experiment-order"><h2>最小实际重建 → 机制集 → 真值辅助参考与小学习</h2>
<table><tr><th>实验</th><th>信息与预算合同</th></tr>
<tr><td>M0 最小实际重建／接口</td><td>普通场景＋受限观测＋当前训练代码；独立核验表面、深度、编辑、effect与实际污染</td></tr>
<tr><td>M1 真值辅助归属参考</td><td>固定真实重建表示，读取额外世界／深度／实例参考；mixed、错深度、无对应和缺测单列</td></tr>
<tr><td>M2 真值辅助表示修复参考</td><td>声明额外几何／材料、容量、新增与优化预算；分别修几何、双侧拆分、effect及组合</td></tr>
<tr><td>M3 候选范围</td><td>固定目标全集、总请求与模型，加入可确认正常背景，报告成本和编辑损伤</td></tr>
<tr><td>M4 小规模联合学习</td><td>机制与隔离成立后才学习，独立场景冻结测试；不先批量造分类数据</td></tr></table>
<p><strong>M1/M2不是数学上界。</strong>真值辅助与完整干净表面参考分栏，逐项列额外信息、可观测域、视角、容量、更新、时间与显存；有限优化不保证优于实际方法，完整参考不能冒充公平方法胜利。测试真值不用于实际方法选参或调优。</p>
<p>开发／训练／校准／测试按物理场景、车辆供体、道路／灯光、轨迹和扰动祖先隔离。<strong>场景1仅开发诊断和迁移推理，不训练新的共享编码／判断／修复／effect模块。</strong>连续状态优化另报test-time adaptation；真实三场景无完整GT，场景1也不是整个方法的未见测试。</p></section>
<section id="v43-environment"><h2>环境进入研究边界：固定分支、四路配对</h2>
<p>C = C<sub>G</sub> + (1−α)E。Actor路径无天空门控；Actor10K冻结原环境和静态。部分黑带来自环境回填，训练成因尚未证明。</p>
<table><tr><th>同相机／时刻／请求</th><th>基线高斯</th><th>待测分解后高斯</th></tr>
<tr><td>原环境E冻结</td><td>原生最终RGB</td><td>完整系统收益／残留</td></tr>
<tr><td>统一线性白底，仅诊断</td><td>原始覆盖／空洞</td><td>分解残影／几何／背景误改</td></tr></table>
<p>每格存浮点C<sub>G</sub>、α、回填、最终RGB，另存E／B与裁剪显示。<strong>环境参数冻结不等于回填贡献冻结。</strong>白洞不是补好道路。当前不重训环境；如成为主要瓶颈，再做固定高斯、仅改环境的单因素对照。</p></section>
<section id="v43-evidence-boundary"><h2>历史诊断保留；运行结果须另有凭据</h2>
<p><a href="gpu_v43/sim_mechanism_20261030/index.html">M0：最小场景原生10K重建与独立编辑核验</a>：16帧r2／r3适配，实际完成原生分支编辑基线与一个留出视角；新V43联合模块未训练，具体结论见独立运行报告。</p>
<p><a href="gpu_v43/sim_m1m2_20261030/index.html">M1/M2：固定归属与分步表示修复</a>已完成首轮3锚点＋留出视角：实际贡献高召回、同候选位置／尺度／拆分，以及同为200步的局部外观控制。仍有明显车形残影；部分删车误差下降，none与横移仍退化。条件贡献不是物理身份GT，不能称理想上界；16帧时序未验收。</p>
<p><a href="gpu_v43/bounded_white_v1/index.html">R2限定白底</a>、<a href="gpu_v43/unknown_white_20261030/index.html">unknown白材质六组204帧</a>、<a href="gpu_v43/editable_ownership_20261030/index.html">三锚点＋200步child refit</a>均不修改。旧短探针未显示编辑分离改善，但不能否定联合学习或建立串行必经顺序。旧49帧／三锚点不冒充新机制集；人工verdict保持空。</p>
<p>独立表面／深度、删车／横移／转向、旧位残留、新位完整度、保留域损伤、效应和任务完成率共同核验。无完整GT的真实数据只报告证据范围，不报告完整真召回。设计状态不代表实验完成。</p></section>
<section id="v43-primary-sources"><h2>直接相关工作与强基线</h2>
<p>继承<a href="https://arxiv.org/html/2503.10437v1">4D LangSplat</a>语义／时间选择和<a href="https://arxiv.org/html/2411.19290v2">TRASE</a>实例编辑强基线；<a href="https://arxiv.org/abs/2403.12722">HUGS</a>与<a href="https://emernerf.github.io/">EmerNeRF</a>提供联合建模依据；<a href="https://docs.typesafe.ai/introduction">Jev</a>仅启发判断／执行分离。原方法和适配版本分别记录，这些依据不等于V43实验已成功。</p>
<p>TRASE已讨论概率对应和联合重建／分割，不能以“联合／概率化”本身主张创新。V43须在相近覆盖和明确预算下证明残影、误改与结构改善。</p></section>
<section id="v43-session-rules"><h2>本对话执行规则</h2><p><strong>禁止自动AutoDL关机；完成或空闲均保留实例，只有用户明确要求才关机。</strong>覆盖旧自动关机约定。subagent仅6.1 sol或6.1 terra。小时级实验先做真实约10分钟smoke，覆盖保存、评价、完整恢复和下一阶段；该生成器不启动GPU任务或历史队列。</p></section>'''


def upgrade_schema(s):
    s = upgrade_schema_r4(s)
    s['schema_version'] = 'v43-joint-editable-decomposition-design-5'
    s['evidence']['unknown_reason_vocabulary'] = list(dict.fromkeys(
        s['evidence']['unknown_reason_vocabulary'] + ['wrong_depth', 'no_correspondence']))
    s['evidence']['record_fields'] = list(dict.fromkeys(
        s['evidence']['record_fields'] + ['input_role', 'partition', 'correspondence_valid_domain_ref']))
    s['evidence']['input_role_vocabulary'] = ['deployment_legal_evidence', 'training_supervision',
                                            'isolated_evaluation_gt', 'truth_assisted_diagnostic']
    s['ground_truth_contract'] = {
        'primary_truth_space': 'independent_simulated_world_surfaces_objects_and_interventions',
        'world_truth_fields': ['surface_id', 'object_id', 'semantic', 'geometry', 'material', 'coordinate_frame',
                               'camera', 'timestamp', 'trajectory', 'lighting', 'visibility', 'intervention'],
        'reference_outputs': ['independent_rgb', 'surface_id_image', 'depth', 'visibility', 'independently_rendered_edit_pairs'],
        'unique_gt_for_every_reconstructed_gaussian': False,
        'original_branch_and_webpage_semantics': 'noisy_evidence_not_ground_truth',
        'alphaT_times_pixel_semantics': 'derived_responsibility_not_gaussian_gt',
        'gaussian_to_surface_correspondence': 'conditional_on_independent_depth_surface_and_visibility_consistency',
        'ambiguities_remain_explicit': ['wrong_depth', 'mixed_support', 'no_correspondence', 'missing_observation'],
        'self_render_or_prediction_is_independent_gt': False,
    }
    s['legal_input_contract'] = {
        'deployment_inputs': ['restricted_rgb', 'sparse_lidar', 'camera_and_time', 'reconstructed_gaussians',
                              'original_branch', 'available_noisy_tracks_masks_semantics', 'render_contribution_evidence'],
        'deployment_forbidden_inputs': ['complete_world_mesh', 'dense_gt_depth', 'world_surface_or_object_gt',
                                       'heldout_edit_reference', 'test_reference_derived_responsibility'],
        'training_gt_access': 'declared_supervision_on_training_partition_only_not_as_deployment_feature',
        'validation_test_gt_access': 'isolated_evaluation_and_explicit_truth_assisted_diagnostic_branches_only',
        'test_gt_for_actual_method_selection_or_adaptation': False,
        'method_input_and_gt_manifests_separate': True,
    }
    s['correspondence'] = {
        'status': 'unknown',
        'vocabulary': ['consistent_single_support', 'wrong_depth', 'mixed_support', 'no_correspondence', 'missing_observation'],
        'world_surface_refs': [], 'independent_depth_refs': [], 'visibility_refs': [],
        'geometry_residual': None, 'valid_domain_ref': None, 'unassigned_reason': None,
        'threshold_contract': 'preregister_scene_scale_sensor_uncertainty_and_visibility_rules',
        'projected_overlap_implies_surface_identity': False,
        'wrong_depth_is_mixed': False,
        'missing_observation_is_no_correspondence': False,
        'force_hard_owner_for_unmatched_gaussian': False,
    }
    s['responsibility'] = {
        'components': [], 'unassigned_mass': None, 'total_contribution_W': None,
        'measurement': 'visible_alpha_times_transmittance_weighted_independent_pixel_surface_semantics',
        'reference_kind': 'derived_observation_domain_responsibility_not_world_or_gaussian_gt',
        'component_value_kind': 'normalized_derived_responsibility_share',
        'required_checks': ['independent_surface_id', 'independent_depth_consistency', 'visibility', 'correspondence_validity'],
        'normalization': 'if W>0: sum(component_shares)+unassigned_mass=1; invalid_or_unmatched_mass_remains_unassigned',
        'zero_contribution': 'unobservable; shares=null, not zero ownership responsibility',
        'multiple_visibility_states': 'separate W and shares per state; never sum into identity probability',
        'domain': {'camera_ids': [], 'timestamps': [], 'pixel_support_ref': None, 'visibility_state': None,
                   'correspondence_valid_domain_ref': None},
        'unassigned_mass_by_reason': {'wrong_depth': None, 'no_correspondence': None, 'missing_observation': None, 'other': None},
        'independent_surface_label_refs': [], 'independent_depth_refs': [], 'rule_version': None,
        'is_ground_truth': False, 'is_identity_probability': False, 'is_spatial_mass_fraction': False,
        'multiple_supports_do_not_require_unique_row_gt': True,
    }
    s['effect']['shadow_contract'] = 'world-bound road receiver geometry plus vehicle-owned shadow effect; receiver does not move with vehicle'
    s['effect']['independent_shadow_reference'] = {
        'body_visibility_and_shadow_casting_controlled_separately': True,
        'vehicle_present_minus_absent_is_pure_shadow_gt': False,
        'object_removal_difference_may_include': ['occlusion', 'reflection', 'indirect_lighting', 'shadow'],
        'required_controls': ['fixed_receiver_geometry_and_material', 'matched_visible_domain', 'explicit_shadow_casting_intervention'],
    }
    s['shadow4'] = [
        {'body': True, 'shadow': True, 'reference_source': 'independent_world_renderer'},
        {'body': False, 'shadow': True, 'reference_source': 'independent_world_renderer_with_body_visibility_control'},
        {'body': True, 'shadow': False, 'reference_source': 'independent_world_renderer_with_cast_shadow_control'},
        {'body': False, 'shadow': False, 'reference_source': 'independent_world_renderer_with_both_controls'},
    ]
    s['truth_assisted_reference_contract'] = {
        'M1': 'truth_assisted_ownership_reference_on_fixed_actually_reconstructed_representation',
        'M2': 'truth_assisted_representation_repair_reference_with_declared_information_capacity_and_optimization_budget',
        'mathematical_upper_bound': False,
        'guaranteed_to_outperform_actual_method': False,
        'required_disclosures': ['extra_truth_fields', 'observable_domain', 'available_views', 'representation_capacity',
                                 'new_gaussians', 'optimization_updates', 'runtime', 'memory'],
        'complete_clean_surface_reference_is_fair_method': False,
        'complete_reference_role': 'independent_edit_target_reference_reported_separately',
        'heldout_truth_tunes_actual_method': False,
    }
    s['data_protocol'] = {
        'primary_route': ['ordinary_existing_or_procedural_vehicle_road_light_world', 'restricted_rgb_sparse_lidar_trajectory',
                          'actual_training_with_current_reconstruction_code', 'observe_natural_contamination', 'v43_legal_inputs',
                          'independent_surface_depth_and_edit_evaluation'],
        'natural_contamination_guaranteed': False,
        'artificial_label_or_parameter_perturbation_role': 'auxiliary_mechanism_control_not_primary_data_route',
        'first_gate': 'one_minimal_scene_actual_reconstruction_and_contamination_audit',
        'then': ['three_to_five_mechanism_scenes', 'small_learning_before_batch_generation'],
        'clean_complete_surfaces_directly_construct_primary_test_gaussians': False,
    }
    s['real_data_reference_contract'] = {
        'scene_count': 3, 'complete_gt_available': False,
        'human_or_subagent_review_is_perfect_gt': False,
        'review_requires': ['scope', 'evidence', 'uncertainty', 'disagreement'],
        'scene1_role': 'development_diagnosis_and_transfer_inference_only',
        'scene1_trains_new_transferable_modules': False,
        'human_verdict': None,
    }
    s['invariants'] += [
        'independent world surface object and intervention truth does not imply a unique GT label per Gaussian',
        'alphaT times pixel semantics is derived responsibility, not Gaussian ground truth',
        'wrong depth, mixed support, no correspondence and missing observation remain distinct',
        'ordinary-world restricted observations pass through actual current-code reconstruction before the primary V43 evaluation',
        'natural contamination is observed and not guaranteed; artificial label noise remains auxiliary',
        'evaluation truth is isolated from deployment inputs and heldout truth never tunes the actual method',
        'truth-assisted M1 and M2 are diagnostic references, not mathematical upper bounds',
        'vehicle-present minus vehicle-absent is not automatically pure shadow ground truth',
        'human and subagent review on real data is incomplete evidence, not perfect GT',
    ]
    s['invariants'] = list(dict.fromkeys(s['invariants']))
    return s


def experiment_plan():
    def task(task_id, purpose, depends=(), **extra):
        return {'id': task_id, 'purpose': purpose, 'depends_on': list(depends),
                'status': 'planned_without_run_receipt', **extra}

    return {
        'task_id': 'WS-V43-WORLD-GT-PROTOCOL-R5-20261030',
        'architecture_revision': 5, 'artifact_kind': 'design_not_job_queue',
        'execution_count_scope': 'document_generator_only_separate_experiment_receipts_required',
        'jobs_started': 0, 'auto_schedule': False, 'new_training_updates': 0, 'new_scene_renders': 0,
        'primary_data_route': ['ordinary_existing_or_procedural_world', 'restricted_rgb_and_sparse_lidar_along_registered_trajectory',
                               'actual_current_code_reconstruction_training', 'natural_contamination_audit',
                               'v43_with_deployment_legal_inputs', 'independent_surface_depth_edit_references'],
        'minimal_first_gate': {'scene_count': 1, 'actual_reconstruction_required': True,
                               'natural_contamination_guaranteed': False,
                               'if_target_error_absent': 'preserve_and_report_zero_or_insufficient_evidence_before_justified_next_step'},
        'mechanism_scenes': {'count_range': [3, 5], 'after_minimal_actual_reconstruction': True,
            'role': 'mechanism_not_generalization_claim',
            'templates': ['single_vehicle_road_shadow', 'overlapping_vehicles', 'wide_view_occlusion_reveal', 'lighting_and_dark_road'],
            'reference_source': 'independent_world_surface_object_and_intervention_renderer',
            'edits': ['none', 'delete', 'translate', 'rotate', 'lighting_change', 'body_shadow_2x2'],
            'primary_pollution_source': 'actual_reconstruction_from_restricted_observations',
            'auxiliary_controls': ['label_dropout', 'wrong_assignment', 'wrong_depth', 'known_component_mixture', 'baked_shadow', 'temporal_identity_error'],
            'auxiliary_controls_replace_natural_pollution_evidence': False,
            'split_keys': ['physical_scene', 'vehicle_donor', 'road_lighting_layout', 'trajectory', 'perturbation_ancestor'],
            'unique_gaussian_class_gt_required': False},
        'tasks': [task('M0', 'one minimal ordinary scene: actual reconstruction, independent GT interface and natural-pollution audit',
                       required_evidence=['observation_manifest', 'reconstruction_config_and_checkpoint', 'loss_and_budget_record',
                                          'independent_surface_depth_edits', 'actual_contamination_or_absence_report']),
                  task('ENV', 'scene1 fixed original environment versus constant-fill paired diagnosis'),
                  task('DATA', 'extend to three-to-five mechanism scenes after the minimal actual-reconstruction gate', ['M0']),
                  task('M1', 'truth-assisted ownership reference on fixed actual reconstruction; minimal scene probe before expansion', ['M0'],
                       mathematical_upper_bound=False),
                  task('M2', 'truth-assisted geometry/split/effect repair reference with disclosed budget; share M1 candidates', ['M0', 'M1'],
                       mathematical_upper_bound=False),
                  task('M3', 'fixed target coverage plus added confirmed background candidates', ['M1']),
                  task('M4', 'small shared-model learning using legal training supervision and isolated tests', ['DATA', 'M1', 'M2'])],
        'parallel_groups': [['M0', 'ENV'], ['M1', 'M2']],
        'ablations': ['feature_threshold', 'probability_decision_only', 'representation_repair_only', 'joint'],
        'coupling_controls': ['sequential_freeze', 'joint_gradients', 'cross_head_stop_gradient'],
        'comparison_contract': {'match_candidate_recall': True, 'match_edit_coverage': True,
            'abstained_targets_remain_in_denominator': True, 'report_capacity_and_compute': True,
            'truth_assisted_M1_M2_are_mathematical_upper_bounds': False,
            'disclose_extra_information': True,
            'disclose_reference_capacity_new_points_updates_runtime_memory': True,
            'heldout_truth_for_actual_method_tuning': False,
            'complete_clean_surface_reference_is_fair_method': False},
        'gt_input_isolation': {'deployment_receives_world_gt': False, 'separate_method_and_evaluation_manifests': True,
            'training_truth_supervision': 'training_partition_only',
            'heldout_edit_pairs': 'independently_generated_evaluation_only',
            'derived_alphaT_semantics_is_gaussian_gt': False,
            'ambiguity_classes': ['wrong_depth', 'mixed_support', 'no_correspondence', 'missing_observation']},
        'shadow': {'receiver_frame': 'world', 'effect_owner': 'vehicle',
            'body_and_cast_shadow_controls_independent': True,
            'vehicle_present_minus_absent_is_pure_shadow_gt': False,
            'check_occlusion_reflection_and_indirect_lighting': True},
        'real_scenes': {'count': 3, 'complete_gt': False, 'human_or_subagent_reference_is_perfect_gt': False},
        'scene1': {'role': 'inference_and_development_diagnostics',
            'train_new_transferable_modules': False, 'test_time_state_optimization_report_separately': True},
        'environment': {'train_environment': False, 'paired_fill': ['frozen_original', 'constant_linear_white'],
            'factorial_gaussians': ['baseline', 'treated'],
            'save_float': ['gaussian_color', 'alpha', 'fill_contribution', 'final_rgb', 'raw_E_or_B']},
        'session_execution': {'autodl_auto_shutdown': False, 'shutdown_requires_explicit_user_request': True,
            'allowed_subagent_models': ['gpt-6.1-sol', 'gpt-6.1-terra'], 'long_run_real_smoke_minutes': 10,
            'smoke_paths': ['forward_backward_finite_loss', 'memory_data_loading', 'checkpoint_save', 'evaluation',
                            'complete_state_restore', 'next_stage_entry'],
            'smoke_and_replay_accounted_separately': True},
        'human_verdict': None,
    }


def upgrade_state(s):
    s = upgrade_state_r4(s)
    s.update(architecture_revision=5,
        scope='world_gt_data_and_evaluation_contract_revision_only',
        first_experiment_order=['one_minimal_ordinary_scene_actual_reconstruction_and_contamination_audit',
            'truth_assisted_ownership_and_representation_references', 'three_to_five_mechanism_scenes',
            'candidate_expansion', 'small_joint_learning_before_batch_generation'],
        training_update_count_scope='revision5_document_generator_only_historical_and_new_run_receipts_separate',
        new_scene_renders_this_revision=0,
        independent_world_gt_protocol_revision=5,
        unique_gaussian_gt_assumed=False,
        natural_contamination_guaranteed=False,
        truth_assisted_references_are_mathematical_upper_bounds=False,
        deployment_legal_inputs_and_evaluation_truth_isolated=True,
        real_three_scenes_complete_gt=False,
        human_or_subagent_review_is_perfect_gt=False,
        minimal_actual_reconstruction_run_receipt='gpu_v43/sim_mechanism_20261030/report_state.json',
        mechanism_dataset_run_receipt='gpu_v43/sim_mechanism_20261030/evidence/dataset_r2/validation.json',
        truth_assisted_reference_run_receipt='gpu_v43/sim_m1m2_20261030/report_state.json',
        experiment_status_source='separate_run_receipts_not_document_generation')
    return s
