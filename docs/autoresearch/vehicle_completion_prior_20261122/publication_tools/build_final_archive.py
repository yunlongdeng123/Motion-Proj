"""显式 essential 正清单与本地轻量源包，不调用 Git。"""
from pathlib import Path
import json
import zipfile
from archive_common import ARCHIVE, dump_json, digest, lightweight_text

PUB=Path(__file__).resolve().parent
WORK=PUB.parent
BASE=WORK.parents[1]
OUT=BASE/'outputs/vehicle_completion_prior_20261122'
REMOTE='/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1'
UPLOAD=REMOTE+'/publication/local_final_inputs_20261122_01'

REMOTE_FILES='''conditioning/prepare_real_inputs.py
conditioning/summary.json
conditioning/verification.json
conditioning/lidar/provenance.json
conditioning/train/086/mapping.json
conditioning/train/110/mapping.json
code/surface_to_gaussians.py
code/surface_conversion_heading_math_check.json
code/bench_surface_student.py
code/bench_surface_student_cpu_check.json
code/run_surface_student.py
code/check_grey_asset_gate.py
code/correct_student_ledger.py
code/audit_registered_shape_cpu.py
code/student_interface.md
student/source/run_surface_student_actual_v1.py
texture_seed101_r1/source/run_vehicle_texturing.executed.py
run_vehicle_texturing.py
prepare_texture_mesh.py
shape/source/propose_point_only.py
shape/seed101_point_only_r1/input_contract_frozen.json
shape/seed101_point_only_r1/actual_point_only_mechanism.json
shape/seed101_point_only_r1/proposal_report.json
shape/resource_ledger.json
semantic/source/generate_amodal_condition.py
semantic/source/propose_semantic_bbox_shape.py
semantic/source/register_semantic_mesh_cpu.py
semantic/source/fix_semantic_heading_cpu.py
semantic/image_point_plan_unused.json
semantic/sd15_seed101_r1/plan.json
semantic/sd15_seed101_r1/receipt.json
semantic/omni_seed101_image_bbox_r1/input_contract_frozen.json
semantic/omni_seed101_image_bbox_r1/proposal_report.json
semantic/omni_seed101_image_bbox_r1/actual_image_bbox_mechanism.json
semantic/resource_ledger.json
semantic/registration_resource_ledger.json
shape_cpu_audit_r1/results/audit.json
semantic_cpu_audit_r1/results/audit.json
semantic_cpu_audit_heading_fixed/results/audit.json
texture_seed101_r1/run_receipt.original.json
texture_seed101_r1/run_receipt.json
prepared_texture/transform.json
prepared_texture/transform_heading_fixed.json
surface_gs_init/surface_conversion.json
grey_gate_v2.json
gate_v2_cpu_proof/proof.json
student_entry_gate.json
student_entry_gate_preview_smoke.json
teacher_inputs_amodal3r/source/prepare_amodal3r_raw110.py
teacher_inputs_amodal3r/raw110/mapping.json
teacher_inputs_amodal3r/raw110/verification.json
student/formal_contract.json
student/surface_supervision_domains.json
student/surface_source.json
student/visible_support.json
student/gradient_isolation.json
student/smoke/proof.json
student/smoke/continuation_numeric.json
student/disk_reload_proof.json
student/delivery/proofs.json
student/delivery/fixed_detail_metrics.json
student/independent_qa.json
student/bench_verification.json
student/resource_ledger_original_inherited.json
student/resource_ledger_corrected.json
student/formal_resource.json
student/smoke_resource.json
student/reload_resource.json
student/assets/joint_fit/scene_binding.json
student/restore_index.json
student/DIAGNOSTIC_RESTORE.md
final_prior_provenance.json
final_prior_archive_receipt.before_metadata_fix.json
final_prior_archive_receipt.corrected.json
final_prior_takeback_manifest.json
final_prior_takeback_manifest.corrected.json'''.splitlines()


def main():
    output=PUB/'local_final_inputs_20261122_01'
    if output.exists():raise FileExistsError(output)
    inventory={r['relative'] for r in json.loads((PUB/'remote_source_inventory.json').read_text())}
    inventory.update(['student/restore_index.json','student/DIAGNOSTIC_RESTORE.md'])
    missing=sorted(set(REMOTE_FILES)-inventory)
    if missing:raise FileNotFoundError('explicit remote sources absent: '+str(missing))
    sources=[];bindings={}
    for rel in REMOTE_FILES:
        destination=('run_code/'+rel[5:] if rel.startswith('code/') else 'run_code/'+rel) if rel.endswith('.py') else rel
        role='run_source' if rel.endswith('.py') else 'receipt' if any(x in rel for x in ['proof','receipt','verification']) else 'metrics' if any(x in rel for x in ['audit.json','metrics.json','independent_qa.json']) else 'record'
        sources.append(dict(source=REMOTE+'/'+rel,destination=ARCHIVE+'/'+destination,role=role))
    def add(relative,path,role):
        data=lightweight_text(path)
        target=output/'tree'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        local=dict(path=str(Path(path).resolve()),sha256=digest(data))
        sources.append(dict(source=UPLOAD+'/tree/'+relative,destination=ARCHIVE+'/'+relative,role=role,local_origin=local))
        bindings[ARCHIVE+'/'+relative]=local
    for name in ['protocol.json','protocol_point_only.json','protocol_semantic_condition.json']:
        add(name,WORK/name,'contract')
    for name in ['teacher_student_protocol_v3.md','teacher_model_audit_20261122.md','mm_trellis_audit.md','prior_review.md']:
        add(name,WORK/name,'record')
    for name in ['build_report.py','assemble_result.py','build_architecture.py','build_architecture_executed_r1.py','link_prior_report.py','qa_report.py']:
        add('report_source/'+name,WORK/name,'report_source')
    add('report_source/report.local.md',WORK/'report.md','report_source')
    add('result.json',WORK/'result.json','report_source')
    add('index.html',OUT/'index.html','report_source')
    for name in ['architecture.svg','architecture_executed_r1.svg']:
        add(name,OUT/name,'architecture_source')
    for name in ['qa_report.json','qa_report_pixel_check.json','student_light_delivery_receipt.json']:
        add('evidence/local/'+name,WORK/name,'receipt')
    add('linked_sources/vehicle_road_joint_20261121/viewer.js',BASE/'work/vehicle_road_joint_20261121/viewer.js','provenance')
    for name in ['archive_common.py','prepare_archive.py','publish_selected.py','validate_archive.py','verify_published.py','verify_takeback.py','build_final_archive.py','README.md','manifest.template.json','scope_contract.json']:
        add('publication_tools/'+name,PUB/name,'publication_tool')
    portable=(WORK/'report.md').read_text(encoding='utf-8')
    portable=portable.replace(str(OUT).replace('\\','/')+'/architecture.svg','architecture.svg')
    portable=portable.replace(str(OUT).replace('\\','/')+'/index.html','index.html')
    generated=output/'derived/report.md';generated.parent.mkdir(parents=True,exist_ok=True);generated.write_text(portable,encoding='utf-8')
    add('report.md',generated,'report_source')
    readme='''# a20 生成表面与材质诊断归档\n\n当前候选为 diagnostic_rejected_geometry，不可晋升。[报告与组件](report.md)、[交互报告源码](index.html)、[实际执行组件](architecture_executed_r1.svg)、[新版教师协议](teacher_student_protocol_v3.md)。主状态和实验索引只链接本目录。\n\n```mermaid\nflowchart LR\n A[raw86/110 真实 RGB / mask / LiDAR] --> B[生成候选 + 表面高斯]\n B --> C[400步可见 RGB 诊断拟合]\n C --> D[固定旧域 bench：当前拒绝]\n A --> E[锁定可信观测表面]\n E --> F[仅缺失区灰模教师]\n F --> G[灰模门先于材质与蒸馏]\n```\n\n`run_code/`保存运行代码与冻结 actual-v1；公共 run_surface_student.py 与 run_vehicle_texturing.py 为未来 v2 gate 入口。`report_source/`保留本地报告生成代码及原始 Markdown，顶层 report.md 只转换本地链接。原始观测、模型、NPZ/PT、PNG/JPG/WebP、PLY/GLB均留在独立交付，未放入Git。\n\n400步只是先验初始化和可见RGB拟合，不是完整教师几何/轮廓/新视角蒸馏。总物理更新409=400正式+9smoke，9已包含2replay；original/corrected ledger与材质元数据成对保留。Amodal3R规范输入仅CPU准备，Amodal3R/MM-TRELLIS/RECAST未运行。\n\n当前公共v2按灰模门有意拒绝此候选。历史只读重载使用冻结student/source/run_surface_student_actual_v1.py、保存的原始entry与compact binding；原scene_binding未改写。详见[恢复说明](student/DIAGNOSTIC_RESTORE.md)与[恢复索引](student/restore_index.json)，既有fresh-process五帧八输出maxdiff=0证明保留。本次归档未再运行GPU重载；历史入口不作为灰模晋升或新训练入口。\n'''
    generated_readme=output/'derived/README.md';generated_readme.write_text(readme,encoding='utf-8');add('README.md',generated_readme,'record')
    local_manifest=output/'derived/local_source_bindings.json';dump_json(local_manifest,dict(source_kind='local report and QA text only',files=bindings));add('provenance/local_source_bindings.json',local_manifest,'provenance')
    for name in ['status_snapshot.md','experiments_append.md','failure_append.md']:
        target=output/'support'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((PUB/name).read_bytes())
    manifest=json.loads((PUB/'manifest.template.json').read_text())
    manifest.update(publication_ready=True,root_report_complete=True,actual_result_kind='diagnostic_rejected_geometry',actual_training_performed=True,sources=sources)
    manifest['status_update'].update(old_source=REMOTE+'/publication/preflight_20261122_01/status_replace_before.md',source=UPLOAD+'/support/status_snapshot.md')
    manifest['experiments_append']['source']=UPLOAD+'/support/experiments_append.md'
    manifest['failure_append']=dict(source=UPLOAD+'/support/failure_append.md',task_marker='VEHICLE-COMPLETION-PRIOR-20261122')
    manifest['commit_message']='feat(scene): archive rejected vehicle surface prior diagnostic\n\nPreserve executed v1, future grey-model gate, fixed-domain metrics and corrected resource records. Record visible-body protection failure and the incomplete distillation loop.'
    dump_json(PUB/'manifest.final.json',manifest)
    dump_json(output/'catalog.json',dict(files=[str(p.relative_to(output)).replace('\\','/') for p in output.rglob('*') if p.is_file()],source_files=len(sources)))
    archive=output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in output.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(output).as_posix())
    dump_json(PUB/'local_upload_receipt.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),sources=len(sources)))
    print(json.dumps(dict(archive=str(archive),sources=len(sources),remote_sources=len(REMOTE_FILES),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
