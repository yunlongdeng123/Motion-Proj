"""Create small publication-review snapshots; never runs Git or model code."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

WS=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-22\065-file-c-users-dengyunlong-documents-2')
R9=WS/'work/old_raw_teacher_joint_20261123_r9';R8=WS/'work/road_vehicle_joint_dggt_20261123_r8';R5=WS/'work/native_source_correction_20261122_r5'
PUB=R9/'publication';PUB.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
selected=[];missing=[]
def snapshot(path,category):
    if not path.is_file():missing.append(str(path));return
    assert path.suffix.lower() in ['.py','.json','.md','.svg','.js']
    assert path.stat().st_size<300_000,str(path)
    source_sha=sha(path);target=PUB/'draft_sources'/category/path.name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and sha(target)!=source_sha:
        before=PUB/'draft_history'/stamp/category/path.name;before.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,before)
    shutil.copy2(path,target);assert sha(path)==sha(target)==source_sha
    selected.append(dict(source=str(path),snapshot=target.relative_to(PUB).as_posix(),bytes=target.stat().st_size,sha256=source_sha))

for filename in ['build_wiring_report.py','wiring_viewer.js','qa_wiring_report.py','joint_fit_adapter.py','dggt_forward_adapter.py','r9_current_architecture_components.svg']:
    snapshot(R8/'report'/filename,'report')
for filename in ['project_old_raw_candidates.py','qualify_raw_road_semantics.py','fuse_old_raw_joint.py','solve_continuous_shared_field.py','audit_r9_source_observations_cpu.py','audit_r9_shared_field_cpu.py','audit_r9_continuous_field_cpu.py','render_r9_continuous_surface_cpu.py','export_r9_teacher_influence_views_cpu.py','approve_root_shared_input.py']:
    snapshot(R9/'code'/filename,'r9_code')
for filename in ['raw_teacher_sources.json','raw_teacher_source_path_resolution.json','brown_mapping_audit.json','visual_source_review.json','fixed_surface_mesh_full_XYZ_audit.json']:
    snapshot(R9/'audit'/filename,'source_audit')
for filename in ['acceptance_contract_v1.json','continuous_field_contract_v2.json']:
    snapshot(R9/filename,'contracts')
for relative in ['raw_road_semantic_v1/failure_receipt.json','raw_road_semantic_v2/receipt.json','world_observations_v2/receipt.json','joint_shared_v1/receipt.json','joint_shared_continuous_v2/receipt.json']:
    snapshot(R9/relative,relative.rsplit('/',1)[0])
for relative in ['joint_shared_v1/numeric_acceptance.json','joint_shared_v1/visual_review.json','joint_shared_v1/source_sampling_qualification_audit.json','joint_shared_continuous_v2/numeric_acceptance.json','joint_shared_continuous_v2/visual_review.json','joint_shared_continuous_v2/root_shared_input_review.json','joint_shared_continuous_v2/continuous_weight_support_audit.json','joint_shared_continuous_v2/teacher_influence_display_receipt.json','joint_shared_continuous_v2/surface_views_receipt.json']:
    snapshot(R9/'cpu_review'/relative,'cpu_review/'+relative.rsplit('/',1)[0])
for relative in ['audit/a20_baseline/baseline_audit.json','audit/mask_coordinate_audit/mask_coordinate_QA.json','audit/baseline_A0_coverage_v1/coverage_occlusion_audit.json','audit/owner_unification_CPU_v1/owner_review_delivery.json','audit/owner_unification_CPU_v1/owner_unification_audit.json','joint_colour_guard_v1/receipt.json']:
    snapshot(R8/relative,'r8/'+relative.rsplit('/',1)[0])
for filename in ['correct_source_provenance.py','source_correction_integrity.json','qa_source_correction.json']:
    snapshot(R5/'report'/filename,'historical_source_correction')
snapshot(WS/'outputs/road_vehicle_joint_dggt_20261123_r8/report_qa/qa_continuous_input_report.json','report_qa')
snapshot(Path(__file__),'publication_builder')

root_review=read(R9/'cpu_review/joint_shared_continuous_v2/root_shared_input_review.json')
assert root_review['decision']=='PASS' and root_review['final_atlas_sha256']=='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'
large=[]
def reference(path,role,expected=None):
    if not path.is_file():missing.append(str(path));return
    actual=sha(path)
    if expected:assert actual==expected
    large.append(dict(role=role,path=str(path),bytes=path.stat().st_size,sha256=actual,selected_for_Git=False,archive_reference_status='pending_final_root_package'))
reference(R9/'joint_shared_continuous_v2/final_atlas.npz','已验收共享表面场',root_review['final_atlas_sha256'])
reference(R9/'joint_shared_v1/final_atlas.npz','初版目视失败共享场','8ad4b9594ef5e14f9f77ebf57a395ebf3da410bb435eb2170f6b2558de9886dd')
reference(WS/'work/road_shared_appearance_20261122_r7/shared_grid_geometry.npz','固定世界曲面几何')
reference(WS/'work/road_shared_appearance_20261122_r7/quilt_seed/atlas_seed.npz','照片参考底纹')
manifest=read(R9/'audit/raw_teacher_sources.json')
for row in manifest['sources']:
    for key,name in [('raw_model','raw_model.png'),('final_native','final_native.png')]:
        reference(R9/f'audit/source_inputs/{row["raw"]:03d}/{name}',f'{row["raw"]}帧{key}；Brown原图',row[key]['sha256'])
for row in root_review['reviewed_images']:
    reference(Path(row['path']),'主审已查看的实际曲面图',row['sha256'])
shutil.copy2(R8/'report/r9_current_architecture_components.svg',PUB/'architecture_components.svg')
inventory=dict(schema='R9.publication_pending_inventory.v1',prepared_UTC=datetime.now(timezone.utc).isoformat(),final_publication=False,Git_operations_executed=False,failure_family='V82-F06',human_verdict=None,problem_solved=None,overall_run_complete=False,
    current_shared_input_sha256=root_review['final_atlas_sha256'],shared_input_acceptance='PASS',joint_formal400_status='pending_actual_result',official_DGGT_status='pending_actual_forward_and_render',per_file_limit_bytes=300_000,
    selected_files=selected,binary_archive_references=large,missing_optional_sources=missing)
(PUB/'selected_inventory.pending.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
text='''道路与动态车辆联合重建：阶段发布草稿

此目录用于最终发布前审阅。联合400步拟合与原版DGGT网络对照仍待真实结果，最终资源账本和资产收据由主任务补齐。

![架构](architecture_components.svg)

| 阶段 | 当前事实 | 状态 |
|---|---|---|
| R7历史 | 顺序回写、联合教师与道路资格修正候选均保留；仅照片参考的有效材质域通过，整片道路未验收 | NOT_ACCEPTED，无新高斯学生 |
| R8对齐基线 | 道路与用户选定A20共享原生光栅；两帧复现最大差0，五帧正确针孔域重评 | 25F0B、0更新 |
| R8归属与保护 | 同实例归属测量2F2B，五视角色彩保护5F5B；RGBA均未改变 | R8累计32F7B、0更新 |
| R9共享初版 | 固定43624重叠纹素，输入校色指标通过，新增条带导致目视失败 | 8ad4失败对照保留，无高斯拟合 |
| R9连续共享场 | 同一原始教师、曲面与观测，改为连续世界RGB残差场；独立数值、五视角目视和主审均通过 | 2eaf共享输入PASS |
| R9联合400步 | 从已对齐道路与A20初始化的固定几何联合RGB拟合 | 待真实结果，当前未计正式更新 |
| 原版DGGT | 独立多帧Transformer无扩散前向与渲染对照 | 待真实结果，不等同于原生高斯RGB优化 |

本轮仅复用两个既有原始教师输出，新扩散0。道路语义共2次冻结推断：第一次保存086预测后只读诊断数组报错，修复后复用086、仅新增110推断。原始失败现场和收据保留。R7历史800步属于旧实验，不计入本轮联合拟合。

连续共享输入SHA256为`2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647`。6407同位实拍纹素硬锁只针对共享图。相似实拍主导层含40850个微弱教师影响尾纹素，最大线性RGB变化0.00020972；教师条件影响来自连续场方程，不是逐点RGB混合α或校准概率。全更新212193纹素的2cm高频相关为0.99999125；固定初版边界低频跳变p95为0.0014863。

历史1792个针孔黑边像素中，1632已有新共享材质且暗像素0，160缺少现有曲面查询支持。R8五帧原C/S及外扩8像素的场景α<.95计数均0。中心、尺度、透明度、相机与刚体轨迹保持固定，原生上下文和其他车辆保护保留。

来源纠正完整保留：R1迁移及质量审计、R2/R3/R4主00/02路块来自加工道路`appearance_fit/asset`，不能作为未编辑原生重建donor迁移证据。R2两个独立`native_static`控制直接来自formal30k静态checkpoint，作为原生静态对照保留。R5原生可见集合与同ID读出诊断保持历史身份，不作为R9的前置质量门槛。

失败家族沿用`V82-F06`，`human_verdict=null`、`problem_solved=null`、`overall_run_complete=false`。共享输入通过不代表未来高斯资产或完整场景恢复已通过。

`selected_inventory.pending.json`记录每个轻量快照的源路径、SHA256与字节数。入选文件仅py/json/md/svg/js，单文件严格小于300000字节；PNG、NPZ和矩阵只保留归档SHA引用。此草稿未执行Git、提交或推送。
'''
(PUB/'README.pending.md').write_text(text,encoding='utf-8')
summary=dict(publication=str(PUB),selected_count=len(selected),selected_bytes=sum(row['bytes'] for row in selected),binary_reference_count=len(large),missing_optional_sources=missing,Git_operations_executed=False,final_publication=False)
print(json.dumps(summary,ensure_ascii=False))
