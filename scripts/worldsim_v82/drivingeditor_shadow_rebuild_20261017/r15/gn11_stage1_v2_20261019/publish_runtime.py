"""Publish the verified v2 startup/runtime milestone without touching old results."""
from pathlib import Path
import datetime,json,shutil

W=Path(__file__).resolve().parent;BASE=W.parent.parent;O=BASE/'outputs/gn11_stage1_v2_20261019'
REPO=W.parent/'r15_closure_run/publish_repo'
DOC=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019')
CODE=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v2_20261019')
def long(p):return Path(chr(92)*2+'?'+chr(92)+str(p.resolve()))
def write(p,t):long(p).parent.mkdir(parents=True,exist_ok=True);long(p).write_text(t,encoding='utf-8')
def copy(a,b):long(b).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(long(a),long(b))
r=json.loads((W/'runtime_progress.json').read_text(encoding='utf-8'));s=r['status']
record=json.loads((O/'task_record.json').read_text(encoding='utf-8'))
record.update(state='FRESH_30K_TRAINING_IN_PROGRESS',updated_utc=r['collected_utc'],
    updates_executed=s['iteration'],actual_GPU_gate_passed=True,worker_pid=s['pid'],
    optimizer_step_counts=s['optimizer_step_counts'],latest_confirmed_checkpoint_steps=r['published_checkpoint_steps'],
    real_inputs_bound=True,road_model_anchor_roles_separately_verified=True,
    monitor_automation='automation-2 ACTIVE every30minutes',quality_accepted=False)
write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
paths=[]
for p in sorted(O.iterdir()):
    if p.is_file() and p.name!='github_publish.json':rel=DOC/p.name;copy(p,REPO/rel);paths.append(rel.as_posix())
for p in sorted((O/'actual_query_projection').glob('*.png')):
    rel=DOC/'actual_query_projection'/p.name;copy(p,REPO/rel);paths.append(rel.as_posix())
for p in sorted(W.iterdir()):
    if p.is_file() and (p.suffix=='.py' or p.name in {'config.json','cpu_test_receipt.json','candidate_manifest.json','evaluation_contract.json'}):
        rel=CODE/p.name;copy(p,REPO/rel);paths.append(rel.as_posix())
for p in sorted((W/'candidate').rglob('*')):
    if p.is_file() and '__pycache__' not in p.parts:rel=CODE/p.relative_to(W);copy(p,REPO/rel);paths.append(rel.as_posix())
status=REPO/'docs/RESEARCH_STATUS.md';old=status.read_text(encoding='utf-8')
marker='## 已完成的原A/B基线与历史证据'
if marker not in old:raise RuntimeError('current status scope changed; review before replacing')
snapshot=f'''# 当前研究状态

更新：{r['collected_utc']}。当前 task=WS-V82-R15-GN11-STAGE1-V2-20261019；用户已开AutoDL，本版fresh第一阶段30K运行中，真实PID{s['pid']}，最新确认{s['iteration']}/30000。质量尚未验收，human_verdict=null。

严格范围：同原A五身份a6/a37/a5/a26/a18、同raw89–137完整49帧与原始输入初始化、seed0，重新第一阶段30000步，第二阶段0。当前static/dynamic实际Adam计数均{s['iteration']}，未加载旧30K模型、未新增重放或消融。原A实际训练源码三文件原始SHA完全匹配后，隔离部署到/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v2_20261019/engine；当前训练入口source_bundle/run_candidate.py，输出run30k，启动2026-10-19T09:00:04+00:00，独占共享GPU锁。运行中源码与目标冻结。

真实输入/CUDA准入完成：相机模型锚帧raw99与SGCM地面平面锚帧raw89分别绑定，平面坐标往返、全部49相机内参/世界变换通过。当前input_bundle_002与gate_002按manifest/hash绑定；固定相机时间-0.48/0.48静态RGB最大差0，梯度组static6/dynamic9/environment1均有限，静态时间梯度无，零更新检查三个优化器实数计数均0。早期锚帧工程诊断仅零更新，未进入训练。

道路query为raw89的a6/a37车身遮挡射线，总15869像素样本，有效道路交点15625。a6：总1149，晚段/全49真实确认均0、供体先验245、未知904；a37：总14720，晚段131–137确认4765、全49确认7787、供体4874、未知2059。晚段包含在全49内；这些是像素射线样本，不是世界面积/干净GT或最终质量。左右供体按用户坐标去标线后，实际几何合格左4249/右1605；5119个先验query使用770个供体样本，独立CPU重建与冻结供体索引及RGB逐项相等，最大RGB差0，未改训练目标。

首个真实5K节点CPU只读通过：静态475740851B/动态303471134B/环境226494608B/训练状态50008B，Adam/模型有限、RNG/SGCM/sampler5000及原生49评估完整；没有做5K GPU恢复或增加优化更新。下一新节点10K/15K/20K/25K/30K按实际发布核验，不重复遍历已验5K。训练完还需当前v2独立完整49帧零更新full/dynamic/static/delete_all/alpha评估及实际全画面质量复核，不能因低loss或训练complete宣称通过。最终evaluator与逐节点CPU audit工具已准备，尚未执行30K评估。

[当前状态、组件图与完整画面道路查询]({DOC.relative_to('docs').as_posix()}/index.html) · [实际运行采样]({DOC.relative_to('docs').as_posix()}/runtime_progress.json) · [5K凭据]({DOC.relative_to('docs').as_posix()}/checkpoint_005000_audit.json)。GPU4080SUPER 32GB，启动时空闲磁盘81.8GB、md0 active/degraded0/idle。automation-2已更新为本版每30分钟跟进，仅阶段变化/失败/需操作/完成通知，不恢复旧A/B队列。全部训练、评估、下载、实际质量报告/Git交付与无worker核对前不关机。failure_ledger_refs=[V82-F06]；本轮为候选真实输入/工程准入与5K耐久证据，不新增失败ID或科学结论。

'''
write(status,snapshot+marker+old.split(marker,1)[1]);paths.append('docs/RESEARCH_STATUS.md')
card=REPO/'docs/research_failures/entries/V82-F06.md';old=card.read_text(encoding='utf-8')
heading='## 2026-10-19：v2真实输入与CUDA零更新通过，fresh30K启动及首5K'
if heading in old:raise RuntimeError('startup milestone already published; use a named subsequent milestone')
entry=f'''

{heading}

用户开机后核对原A三份实际源码原始字节一致，在独立目录部署当前v2并于2026-10-19T09:00:04+00:00启动fresh30K，PID2145，第二阶段0。相机模型raw99与SGCM道路raw89的锚点角色独立绑定并做平面往返核验；全部49相机记录内参和查询标定差0，实际模型世界变换验证通过。当前gate_002的固定相机跨时间静态RGB最大差0，梯度全有限、静态时间梯度无、三个优化器更新计数均0；当前训练只用input_bundle_002与该gate精确绑定。前置工程诊断不计训练更新，未加载旧模型或重放。

真实道路查询分母15869个raw89车身像素射线样本。a6在当前严格条件下晚段/全49确认均0，先验245、未知904；a37晚段确认4765、全49确认7787，先验4874、未知2059。它们不是均匀世界面积、未见道路真值或模型效果结论。实际供体几何支撑左4249/右1605，5119个先验query使用770个不同供体样本；独立CPU重建来源索引/RGB与冻结目标相等、RGB最大差0，没有改运行中输入。报告分别显示query原帧来源类别、后续帧当前参与融合/仅其他帧确认/先验/未知，均完整480×320。

首个5000完整native节点CPU只读核验：static475740851B、dynamic303471134B、env226494608B、training_state50008B，模型/Adam有限，RNG/SGCM/sampler5000/原生49评估齐全，未做5K GPU恢复或增加优化更新。当前实际最新{s['iteration']}步并继续至30K；旧A/B完成状态不当本版完成，最终49零更新删除/质量复核仍待执行。human_verdict=null。

[真实输入、CUDA、首5K和全幅查询证据](../../{DOC.relative_to('docs').as_posix()}/index.html)。failure_ledger_delta=real_input_GPU_admission_and_5K_durability_no_new_scientific_verdict。
'''
write(card,old+entry);paths.append('docs/research_failures/entries/V82-F06.md')
(W/'publish_runtime_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(paths),iteration=s['iteration'],training=True,quality_accepted=False)))
