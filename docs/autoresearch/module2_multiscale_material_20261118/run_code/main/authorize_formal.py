from pathlib import Path
import json,time,shutil

P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')
smoke_path=P/'lifecycle_final_readonly_r1/verified_smoke_receipt.json'
s=json.loads(smoke_path.read_text())
assert s['completed'] is True and s['elapsed_seconds']>=600
assert s['formal_updates']==0 and s['optimizer_updates']<=360 and s['all_K_checks_exact']
assert s['all_geometry_and_coverage_frozen'] and s['formal_must_start_fresh_zero_fields']
for check in s['checks']:
    if 'maximum_state_difference' in check:assert check['maximum_state_difference']<=1e-6
    if 'full_model_optimizer_RNG_sampler_exact' in check:assert check['full_model_optimizer_RNG_sampler_exact'] is True
reload=next(x for x in s['checks'] if x['test']=='authoritative_field_reload')
assert reload['ci_max']==0 and all(x['field_native_RGB_max']==0 and x['RGB8_max']==0 for x in reload['rows'])
assert json.loads((P/'cpu_numeric_r1/receipt.json').read_text())['numeric_gate_pass'] is True
p=P/'formal_launch_contract.json'
if p.exists():raise FileExistsError('唯一正式启动登记已存在')
d=dict(schema='module2_shared_material_formal_launch.v1',task_id='MODULE2-MULTISCALE-MATERIAL-20261118',
 formal_authorized=True,physical_updates=1400,each_method_updates=800,common_prefix_updates=200,
 smoke_receipt=str(smoke_path),minimum_smoke_seconds=600,
 actual_verified_smoke_seconds=s['elapsed_seconds'],smoke_updates=s['optimizer_updates'],prior_failed_updates=4,
 fresh_initial_field=str(P/'cpu_exact_r1/field_initial.pt'),smoke_model_must_not_initialize_formal=True,
 scope=str(P/'method_scope_r1/scope.npz'),run_output=str(P/'formal_r1'),
 code='code/run_material_pair.py',GPU_owner='road_asset',
 registered_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 no_automatic_budget_extension=True,read_only_controls_and_final_verification='part of same planned queue; no geometry/alpha or generator training',
 matched_independent_RGB800='separate matched_rgb_control_contract.json; not counted in this1400; GPU launch will be separately sequenced',
 numeric_scope='GH15 representative primary-source test qualified; lowvalidmass/outlier sensitivity retained in qa/integrator_difference_scope.json, no full-domain shrink',
 GPU_keep_on=True,production_promoted=False,human_verdict=None)
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
lease=json.loads((P/'gpu_lease.json').read_text());assert lease['owner']=='road_asset'
lease['scope']='formal shared material unique1400 physical updates, exactA200 common prefix withL800/ABC800, required checkpoints/evaluation/requery'
lease['active']=True;(P/'gpu_lease.json').write_text(json.dumps(lease,indent=2)+'\n')
R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
(R/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

更新：2026-11-18（Asia/Singapore）。

`MODULE2-MULTISCALE-MATERIAL-20261118 / road_multiscale_material_20261118_r1` 正式共享材质训练已获执行资格。固定修复覆盖资产，以真实世界材质图A0、12cm低频与2cm受许可高频场构成颜色；真实chart反查与物理Σ过滤替代中心点取色，几何/覆盖/车辆/静态/相机/生成教师全部冻结。GPU保持开机。

已完成实际617.32s生命周期：360工程更新、之前恢复接线失败4更新另记，完整模型/Adam/RNG/sampler恢复exact、下一步误差1.86e-9，全部34197300个物理worldquad重构与serialized查询一致，权威field重查/实际native/派生GS缓存和五帧K77266、alpha均通过。Adam CPU-step别名污染已以deepcopy恢复修复，未放宽门，正式fresh0不复用smoke参数。

正式唯一共享方法物理1400更新：共同低频A200，再分支L续600与B高频300/C联合300，每方法有效800步且相同采样序列。另登记同初值/84266范围/S3131/q1的独立RGB800公平对照，未自动并发GPU。原199BUILD继承，当前86/110训练、134/179/198诊断，不是盲测或199帧扩展。

TRELLIS.2固定2.56m道路mesh的一次新推理完成，UV拆缝不改几何，参考仅合法86/110；没有模型微调。高分辨baseColor的GH积分在边界不收敛，已用物理Σ midpoint数值修复并保留同积分A0参照；待统一nativeGS列与评价。不能将积分失败当模型质量否定，也不主张整街先验覆盖。

当前S完整3187、充分覆盖3131(含新909)、薄56，原D38416及K77266分母保持。GH7频响失败与全局affine坐标误差已独立记录，GH15/精确chart资格通过；低source-kernel-mass误差另列，未因此缩训练S。默认场景未替换，无电源队列。human_verdict/problem_solved=null；failure_ledger_refs=[V82-F06]。
''',encoding='utf-8')
print(json.dumps(dict(formal_authorized=True,physical_updates=1400,smoke_seconds=s['elapsed_seconds'],GPU_keep_on=True)))
