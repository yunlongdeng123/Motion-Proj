"""按显式arm:step发布已经独立CPU核验的新节点；不运行远端任务。"""
from pathlib import Path
import argparse,datetime,json,re,shutil
W=Path(__file__).resolve().parent;BASE=W.parent.parent
O=BASE/'outputs/gn11_stage1_v2_20261019';REPO=W.parent/'r15_closure_run/publish_repo'
DOC=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019')
CODE=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v2_20261019')
ARMS=['real_only','prior_only','ownership_only']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def long(p):return Path(chr(92)*2+'?'+chr(92)+str(p.resolve()))
def write(p,t):long(p).write_text(t,encoding='utf-8')
def copy(a,b):long(b).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(long(a),long(b))

def publish(requested):
    nodes=[]
    for value in requested:
        arm,raw=value.rsplit(':',1);step=int(raw)
        if arm not in ARMS or step not in range(5000,30001,5000):raise ValueError('explicit legal arm:5K node required')
        nodes.append((arm,step))
    if len(set(nodes))!=len(nodes):raise ValueError('duplicate requested node')
    nodes=sorted(nodes);r=read(W/'ablation_runtime.json')
    if 'FAILED' in r['registry']['status'] or any('failed' in (a['status'] or {}).get('status','') for a in r['arms'].values()):
        raise RuntimeError('failure requires its own evidence/status update, not a normal node publication')
    card=REPO/'docs/research_failures/entries/V82-F06.md';old=card.read_text(encoding='utf-8')
    label='、'.join(f'{arm} {step//1000}K' for arm,step in nodes)
    day=datetime.datetime.fromisoformat(r['collected_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()
    heading='## '+day+'：消融节点 '+label+' CPU只读核验'
    if heading in old:raise RuntimeError('named node batch already published')
    evidence=[];newfiles=[]
    for arm,step in nodes:
        a=r['arms'][arm];name=f'{arm}_checkpoint_{step:06d}_audit.json';p=O/'ablations'/name
        q=read(p)
        if '/'+name in old:raise RuntimeError('individual node already published: '+name)
        if (a['manifest']['variant']!=arm or step not in a['published_steps'] or step not in a['audited_steps'] or
            q['status']!='PASSED_CPU_NODE_DURABILITY' or q['step']!=step or q['sampler_steps']!=step or
            q['GPU_restore_performed'] or q['training_updates']!=0 or
            not all(q[k] for k in ['SGCM_present','RNG_present','native49_present'])):
            raise RuntimeError('actual arm/node CPU receipt contract differs')
        filenames=[f'chkpnt{step}.pth',f'dynamic_chkpnt{step}.pth',f'env_light_chkpnt{step}.pth',f'training_state{step}.pth']
        sizes='/'.join(str(q['files'][n]['bytes'])+'B' for n in filenames)
        evidence.append(f'{arm}真实{step}完整native节点：四组件（静态/动态/环境/训练状态）{sizes}；模型与Adam有限，RNG/SGCM/sampler{step}及原生49齐全。[独立凭据](../../{DOC.relative_to("docs").as_posix()}/ablations/{name})。')
        newfiles.append(name)
    active=[];queued=[];completed=['完整版30K']
    for name,a in r['arms'].items():
        s=a['status']
        if s:
            if any(s['optimizer_step_counts'][m]!=s['iteration'] for m in ['static','dynamic']):raise RuntimeError('actual update counts differ')
            if s['status']=='complete':
                if s['iteration']!=30000 or any(s['optimizer_step_counts'][m]!=30000 for m in ['static','dynamic','environment']):
                    raise RuntimeError('complete training requires actual 30K counters')
                completed.append(name+' 30K')
            else:active.append(f'{name} PID{s["pid"]} {s["iteration"]}/30000（{s["status"]}）')
        else:queued.append(name)
    live='；'.join(active) or '无在训消融'
    queue='、'.join(queued) or '无'
    currentfile=REPO/'docs/RESEARCH_STATUS.md';s=currentfile.read_text(encoding='utf-8')
    marker='## 已完成的原A/B基线与历史证据';current,history=s.split(marker,1)
    current=re.sub(r'更新：[^。]+。',f'更新：{r["collected_utc"]}。',current,count=1)
    paragraphs=current.split('\n\n')
    finished='、'.join(completed)+'；仅表示训练及随节点原生评估结束，最终独立零更新49和质量对照仍待交付。'
    if not any(p.startswith('已完成训练：') for p in paragraphs):paragraphs.insert(2,'已完成训练：'+finished)
    for i,p in enumerate(paragraphs):
        if p.startswith('已完成训练：'):
            paragraphs[i]='已完成训练：'+finished
        elif p.startswith('当前实际在训：'):
            paragraphs[i]=f'当前实际在训：{live}。待执行：{queue}。实际static/dynamic更新计数与各自进度一致；controller4120继续限制总并发最多2（gate也占槽）。以各自status解释实际阶段，不把静态队列名或guardian存活当训练推进；不得重复启动或恢复旧A/B队列。'
        elif p.startswith('保存证据：'):
            audited='；'.join(name+' '+('/'.join(str(n//1000)+'K' for n in a['audited_steps']) or '尚无已审节点') for name,a in r['arms'].items())
            paragraphs[i]='保存证据：完整版5/10/15/20/25/30K六节点此前均CPU只读通过，不重复核验。各消融独立已审节点：'+audited+'。模型/Adam有限，RNG/SGCM/sampler和各自原生49齐全；CPU核验无GPU恢复或新增更新，不代表视觉质量通过，不能跨模型混用凭据。'
        elif p.startswith('资源采样：'):
            paragraphs[i]=f'资源采样：{r["gpu"]}；md0 {r["md0"]["array_state"]}/degraded{r["md0"]["degraded"]}/sync{r["md0"]["sync_action"]}。磁盘空{r["disk_free_bytes"]/1e9:.3f}GB，剩余资产估{r["disk_required_estimate_bytes"]/1e9:.3f}GB（含8GB余量，点数预测不是严格上界）。controller登记状态{r["registry"]["status"]}；全部训练退出并释放GPU锁前不得启动最终评估，全部对照交付前不关机。automation-2 ACTIVE每30分钟，普通无变化安静。'
    record=read(O/'task_record.json');record['updated_utc']=r['collected_utc']
    record['state']='ALL_FOUR_TRAININGS_COMPLETE_FINAL_DELIVERY_PENDING' if all((a['status'] or {}).get('status')=='complete' for a in r['arms'].values()) else 'MAIN_30K_TRAINING_COMPLETE_ABLATIONS_IN_PROGRESS'
    record['failure_ledger_delta']='named_ablation_CPU_nodes_no_quality_verdict'
    record['resources']['disk_free_bytes_at_snapshot']=r['disk_free_bytes']
    for name,a in r['arms'].items():
        status=a['status'] or {}
        record['ablation_runs'][name].update(status=status.get('status','queued'),updates_executed=status.get('iteration',0),
            actual_optimizer_step_counts=status.get('optimizer_step_counts'),audited_checkpoint_steps=a['audited_steps'],
            worker_pid=(a['train_launch'] or {}).get('pid'))
    write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
    protocol_files=[];protocol_evidence=[]
    for arm,a in r['arms'].items():
        name=arm+'_protocol_receipt.json';p=O/'ablations'/name
        if not p.exists():continue
        protocol=read(p)
        if protocol['variant']!=arm or protocol['first_steps']!=100 or not protocol['frame_and_time_equal'] or not protocol['all_optimizer_learning_rates_equal'] or protocol['training_updates']!=0:
            raise RuntimeError('invalid actual first100 protocol receipt')
        protocol_files.append(name)
        gate=a['gate'];checks=gate['actual_checks']
        protocol_evidence.append(f'{arm}实际train启动{a["train_launch"]["started_utc"]}，PID{a["train_launch"]["pid"]}；CUDA准入0更新、固定相机静态时间RGB差{checks["static_fixed_camera_max_RGB_delta"]}、有限梯度组{checks["finite_gradient_parameter_groups"]}，实际道路权重{checks["actual_road_weights"]}。前100条日志的选帧/时间扰动和已记录静态、动态学习率与完整版相同；环境学习率不在这份日志中，不能据此声称全部优化器实测或旧初始张量逐位相同。[协议凭据](../../{DOC.relative_to("docs").as_posix()}/ablations/{name})。')
    entry='\n\n'+heading+'\n\n'+f'采样{r["collected_utc"]}：{live}；排队{queue}。已完成训练：{finished} 未改训练目标、源码、输入或预算，没有重放/重启。\n\n'+'\n\n'.join(evidence)+'\n\n'+'\n\n'.join(protocol_evidence)+'\n\n以上均CPU只读，无节点GPU恢复或新增训练更新，不能因保存完整或低loss宣称质量通过。四组最终独立zero-update49与全画面质量对照待全部训练结束后执行；当前不关机。human_verdict=null；failure_ledger_delta=named_ablation_CPU_nodes_no_quality_verdict。\n'
    write(currentfile,'\n\n'.join(paragraphs)+marker+history);write(card,old+entry)
    paths=['docs/RESEARCH_STATUS.md','docs/research_failures/entries/V82-F06.md']
    for name in ['index.html','runtime_progress.json','local_delivery_checks.json','task_record.json','candidate_source.zip']:
        rel=DOC/name;copy(O/name,REPO/rel);paths.append(rel.as_posix())
    for name in ['index.html','runtime.json','local_delivery_checks.json',*newfiles,*protocol_files]:
        rel=DOC/'ablations'/name;copy(O/'ablations'/name,REPO/rel);paths.append(rel.as_posix())
    for name in ['publish_ablation_nodes.py','verify_ablation_protocol.py','build_ablation_report.py']:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    key='_'.join(f'{arm}_{step:06d}' for arm,step in nodes)
    (W/f'publish_nodes_{key}_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps(dict(paths=len(paths),nodes=nodes,active=live,queued=queue)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nodes',nargs='+',required=True);a=p.parse_args();publish(a.nodes)
