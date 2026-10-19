"""发布主30K完成、供体组实际启动及各自新节点，绝不触发GPU工作。"""
from pathlib import Path
import json,shutil
W=Path(__file__).resolve().parent;BASE=W.parent.parent
O=BASE/'outputs/gn11_stage1_v2_20261019';REPO=W.parent/'r15_closure_run/publish_repo'
DOC=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019')
CODE=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v2_20261019')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def long(p):return Path(chr(92)*2+'?'+chr(92)+str(p.resolve()))
def write(p,t):long(p).write_text(t,encoding='utf-8')
def copy(a,b):long(b).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(long(a),long(b))

def main():
    r=read(W/'runtime_progress.json');a=read(W/'ablation_runtime.json');s=r['status']
    real=a['arms']['real_only'];prior=a['arms']['prior_only']
    if s['status']!='complete' or any(s['optimizer_step_counts'][m]!=30000 for m in ['static','dynamic']):
        raise RuntimeError('actual complete30K and counters required')
    if r['audited_steps']!=list(range(5000,30001,5000)):raise RuntimeError('six real CPU nodes required')
    if not prior['gate'] or not prior['status'] or prior['status']['iteration']<100:raise RuntimeError('prior arm not actually admitted/training')
    if prior['gate']['optimizer_updates']!=0:raise RuntimeError('zero-update gate required')
    if not all(step in real['audited_steps'] for step in [5000,10000]):raise RuntimeError('real arm5/10K not audited')
    for arm in [real,prior]:
        q=arm['status']
        if any(q['optimizer_step_counts'][m]!=q['iteration'] for m in ['static','dynamic']):raise RuntimeError('actual counters differ')
    card=REPO/'docs/research_failures/entries/V82-F06.md';old=card.read_text(encoding='utf-8')
    heading='## 2026-10-19：v2主30K完成，真实/供体两组消融接续并行'
    if heading in old:raise RuntimeError('transition already published; use later actual milestone')
    record=read(O/'task_record.json')
    record.update(state='MAIN_30K_TRAINING_COMPLETE_ABLATIONS_IN_PROGRESS',updated_utc=r['collected_utc'],
        updates_executed=30000,optimizer_step_counts=s['optimizer_step_counts'],
        latest_confirmed_checkpoint_steps=r['published_checkpoint_steps'],audited_checkpoint_steps=r['audited_steps'],
        main_worker_exited=r['process'].get('exists') is False,final_zero_update49_pending=True,quality_accepted=False,
        failure_ledger_delta='main30K_durable_prior_admitted_no_quality_verdict')
    record['resources']['disk_free_bytes_at_snapshot']=a['disk_free_bytes']
    for name,arm in a['arms'].items():
        q=arm['status'] or {}
        record['ablation_runs'][name].update(status=q.get('status','queued'),updates_executed=q.get('iteration',0),
            actual_optimizer_step_counts=q.get('optimizer_step_counts'),actual_zero_update_gate_passed=bool(arm['gate']),
            worker_pid=(arm['train_launch'] or {}).get('pid'),audited_checkpoint_steps=arm['audited_steps'])
    write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
    statusfile=REPO/'docs/RESEARCH_STATUS.md';status=statusfile.read_text(encoding='utf-8')
    marker='## 已完成的原A/B基线与历史证据';current,history=status.split(marker,1)
    stable='\n\n'.join(p for p in current.split('\n\n') if p.startswith(('真实输入/CUDA准入完成：','道路query为')))
    current=f'''# 当前研究状态

更新：{a['collected_utc']}。task=WS-V82-R15-GN11-STAGE1-V2-20261019及三组ROAD-ABLATIONS。完整版已真实complete30000/30000，static/dynamic/environment实际optimizer.step各30000，原PID2145已退出。原生随保存的49评估完成，但四组最终独立零更新49与实际视觉质量均未交付，human_verdict=null。

当前实际在训：real_only PID{real['status']['pid']} {real['status']['iteration']}/30000；prior_only PID{prior['status']['pid']} {prior['status']['iteration']}/30000。两者实际static/dynamic更新计数与各自进度一致。ownership_only仍排队，controller4120继续限制总并发最多2（gate也占槽），不得重复启动或恢复旧A/B队列。prior_only已通过真实CUDA零更新gate，静态固定相机跨时间RGB最大差0、梯度6/9/1有限、优化器计数0，真实道路RGB/深度权重0、供体权重.1/.01；其前100选帧/时间扰动/全部优化器LR与完整版实际日志一致。

保存证据：完整版5/10/15/20/25/30K六个节点已逐一CPU只读通过，终点四组件569273779/786948958/226494608/49944B，模型/Adam有限、RNG/SGCM/sampler30000/native49齐全。real_only独立5K、10K凭据通过，不能拿主实验receipt替代。所有节点核验均无GPU恢复或新增优化更新，不代表视觉通过；已验节点不重复遍历。

严格范围：四组各fresh第一阶段30K、同原A五身份a6/a37/a5/a26/a18、原49帧与输入初始化协议、seed0，第二阶段0。消融保持归属修复，分别关闭真实、先验或全部道路目标；输入query/provenance/目标不改，UNKNOWN无道路监督。未修改已加载训练源码，未新增重放。全部四组训练、节点核验、最终零更新49、下载、完整画面对照与Git完成后才按授权关机；当前不能启动GPU评估或关机。

资源采样：{a['gpu']}；md0 {a['md0']['array_state']}/degraded{a['md0']['degraded']}/sync{a['md0']['sync_action']}。磁盘空{a['disk_free_bytes']/1e9:.3f}GB，剩余资产估{a['disk_required_estimate_bytes']/1e9:.3f}GB（含8GB共享余量，点数预测不是严格上界）。controller在主退出后持全局GPU锁，评估必须等所有训练退出并释放锁。automation-2 ACTIVE每30分钟，普通无变化安静。

{stable}

[完整版状态、组件和保存证据]({DOC.relative_to('docs').as_posix()}/index.html) · [四组实际队列与独立节点]({DOC.relative_to('docs').as_posix()}/ablations/index.html)。全部为训练重建视图与工程/保存证据，质量复核尚未完成。failure_ledger_refs=[V82-F06]，不新增失败ID或效果结论。

'''
    evidence=f'''{heading}

{r['collected_utc']}确认完整版实际complete30K，static/dynamic/environment实际optimizer.step均30000，原PID2145已退出CUDA与进程。新增25K、30K节点CPU只读通过（终点四组件569273779/786948958/226494608/49944B），加上先前5/10/15/20K，六节点完整；Adam/模型有限、RNG/SGCM/sampler30000和原生49完整。终点CPU检查不代表CUDA恢复或视觉通过，最终独立删除/alpha49评估尚未执行。

调度器4120按已授权两路上限自动接续prior_only，真实启动UTC={prior['train_launch']['started_utc']}，PID{prior['status']['pid']}，当前{prior['status']['iteration']}步；real_only PID{real['status']['pid']}当前{real['status']['iteration']}步，ownership_only继续排队。prior_only实际零更新gate确认权重observed_rgb/depth=0、prior_rgb=.1/prior_depth=.01；静态固定相机跨时间RGB差0，梯度6/9/1有限，更新0。新增前100选帧、时间扰动和所有优化器LR与完整版真实日志相同。没有新重放或修改运行中源码/目标。

real_only独立5K组件480690803/266601246/226494608/50072B、10K组件520838259/824419678/226494608/50008B均CPU只读通过，Adam/RNG/SGCM/sampler和各自原生49完整，无GPU恢复/训练更新。不能混用主实验节点或由低loss宣布分离通过。当前GPU、磁盘、md0和完整四组对照合同见报告；全部训练与对照交付前不关机。human_verdict=null。

[主30K、各组独立节点与实际队列](../../{DOC.relative_to('docs').as_posix()}/ablations/index.html)。failure_ledger_delta=main30K_and_real_only_nodes_durable_prior_admitted_no_quality_verdict。
'''
    write(statusfile,current+marker+history);write(card,old+'\n\n'+evidence)
    paths=['docs/RESEARCH_STATUS.md','docs/research_failures/entries/V82-F06.md']
    for name in ['index.html','runtime_progress.json','local_delivery_checks.json','task_record.json','candidate_source.zip','checkpoint_025000_audit.json','checkpoint_030000_audit.json']:
        rel=DOC/name;copy(O/name,REPO/rel);paths.append(rel.as_posix())
    for p in (O/'ablations').iterdir():
        if p.is_file() and p.name!='github_publish.json':
            rel=DOC/'ablations'/p.name;copy(p,REPO/rel);paths.append(rel.as_posix())
    for name in ['enrich_runtime_report.py','build_ablation_report.py','publish_main30k_transition.py']:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    (W/'publish_main30k_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(paths),main_complete=True,real_only=real['status']['iteration'],prior_only=prior['status']['iteration'],quality_accepted=False)))

if __name__=='__main__':main()
