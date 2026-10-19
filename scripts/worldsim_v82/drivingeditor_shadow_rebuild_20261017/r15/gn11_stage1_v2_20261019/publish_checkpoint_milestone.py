"""发布一个已核验的 v2 保存节点，不修改远端训练。"""
from pathlib import Path
import argparse,json,re,shutil

W=Path(__file__).resolve().parent
BASE=W.parent.parent
O=BASE/'outputs/gn11_stage1_v2_20261019'
REPO=W.parent/'r15_closure_run/publish_repo'
DOC=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019')
CODE=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v2_20261019')

def long(p):return Path(chr(92)*2+'?'+chr(92)+str(p.resolve()))
def write(p,t):long(p).parent.mkdir(parents=True,exist_ok=True);long(p).write_text(t,encoding='utf-8')
def copy(a,b):long(b).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(long(a),long(b))

def publish(step):
    if step not in range(5000,30001,5000):raise ValueError('explicit5K node required')
    r=json.loads((W/'runtime_progress.json').read_text(encoding='utf-8'));s=r['status']
    if s['status']!='training' or s['iteration']<step:raise RuntimeError('training milestone snapshot required')
    if any(s['optimizer_step_counts'][m]!=s['iteration'] for m in ['static','dynamic']):
        raise RuntimeError('actual completed update counters differ')
    if step not in r['published_checkpoint_steps'] or step not in r['audited_steps']:
        raise RuntimeError('actual published and audited node required')
    receipt=f'checkpoint_{step:06d}_audit.json'
    a=json.loads((O/receipt).read_text(encoding='utf-8'))
    if a['status']!='PASSED_CPU_NODE_DURABILITY' or a['step']!=step or a['sampler_steps']!=step:
        raise RuntimeError('receipt step differs')
    if a['GPU_restore_performed'] or a['training_updates']!=0:raise RuntimeError('CPU-only receipt required')
    card=REPO/'docs/research_failures/entries/V82-F06.md'
    oldcard=card.read_text(encoding='utf-8')
    heading=f'## 2026-10-19：v2 {step//1000}K完整保存节点CPU只读核验'
    if heading in oldcard:raise RuntimeError('named milestone already published; do not append twice')
    statusfile=REPO/'docs/RESEARCH_STATUS.md'
    status=statusfile.read_text(encoding='utf-8')
    marker='## 已完成的原A/B基线与历史证据'
    if marker not in status:raise RuntimeError('status scope changed')
    current,history=status.split(marker,1)
    current=re.sub(r'更新：[^。]+。',f'更新：{r["collected_utc"]}。',current,count=1)
    current=re.sub(r'最新确认\d+/30000',f'最新确认{s["iteration"]}/30000',current,count=1)
    current=re.sub(r'当前static/dynamic实际Adam计数均\d+',f'当前static/dynamic实际Adam计数均{s["iteration"]}',current,count=1)
    names=[f'chkpnt{step}.pth',f'dynamic_chkpnt{step}.pth',f'env_light_chkpnt{step}.pth',f'training_state{step}.pth']
    sizes='/'.join(str(a['files'][n]['bytes'])+'B' for n in names)
    audited='/'.join(f'{n//1000}K' for n in r['audited_steps'])
    pending='/'.join(f'{n//1000}K' for n in range(step+5000,30001,5000)) or '无后续训练保存节点'
    node=f'当前{audited}节点已有独立CPU只读核验凭据；最新{step//1000}K四组件（静态/动态/环境/训练状态）{sizes}，模型与Adam有限、RNG/SGCM/sampler{step}/原生49评估齐全。没有节点GPU恢复或新增优化更新，不代表视觉质量通过。下一新节点{pending}按实际发布核验，不重复重验已通过节点。最终v2独立完整49帧零更新full/dynamic/static/delete_all/alpha评估和实际全画面质量复核仍待30K后执行。'
    paragraphs=current.split('\n\n')
    matches=[i for i,p in enumerate(paragraphs) if p.startswith(('首个真实5K节点CPU只读通过','当前5K')) and '节点' in p]
    if len(matches)!=1:raise RuntimeError('checkpoint status paragraph changed; inspect before publishing')
    paragraphs[matches[0]]=node
    current='\n\n'.join(paragraphs)
    current=re.sub(r' · \[(?:最新)?\d+K凭据\]\([^)]*\)',f' · [最新{step//1000}K凭据]({DOC.relative_to("docs").as_posix()}/{receipt})',current,count=1)
    current=re.sub(r'(?:启动时空闲磁盘|本轮采样磁盘空闲)[^。]+',f'本轮采样磁盘空闲{r["disk_free_bytes"]/1e9:.3f}GB、md0 {r["md0"]["array_state"]}/degraded{r["md0"]["degraded"]}/{r["md0"]["sync_action"]}',current,count=1)
    current=re.sub(r'(?:候选)?真实输入/工程准入与[^，。]+证据',f'真实输入/工程准入与{audited}保存证据',current,count=1)
    record=json.loads((O/'task_record.json').read_text(encoding='utf-8'))
    record.update(updated_utc=r['collected_utc'],updates_executed=s['iteration'],
        optimizer_step_counts=s['optimizer_step_counts'],latest_confirmed_checkpoint_steps=r['published_checkpoint_steps'],
        audited_checkpoint_steps=r['audited_steps'],quality_accepted=False,
        failure_ledger_delta=f'CPU_durability_through_{step}_no_new_scientific_verdict')
    record['input_roles']['RGB_masks_initialization']='same original A49 inputs and seed0 protocol verified; no old initial GPU tensor snapshot available'
    record['resources'].update(GPU_memory_admission_pending=False,disk_admission_pending=False,
        GPU=s['GPU'],disk_free_bytes_at_snapshot=r['disk_free_bytes'])
    write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
    entry=f'\n\n{heading}\n\n最新采样{r["collected_utc"]}：同一PID{s["pid"]}推进到{s["iteration"]}/30000，static/dynamic实际optimizer.step计数各{s["optimizer_step_counts"]["static"]}。md0 {r["md0"]["array_state"]}/degraded{r["md0"]["degraded"]}/sync {r["md0"]["sync_action"]}，磁盘空闲{r["disk_free_bytes"]/1e9:.3f}GB；训练目标、输入、预算与已加载源码未改。\n\n真实{step}完整native保存节点CPU只读通过：四组件（静态/动态/环境/训练状态）{sizes}，模型/Adam张量有限，RNG/SGCM/sampler{step}/原生49齐全。此核验没有GPU恢复或新增更新，不能当视觉通过。此前已验节点复用凭据，最终30K零更新49及完整画面质量仍待执行。human_verdict=null。\n\n[节点凭据](../../{DOC.relative_to("docs").as_posix()}/{receipt}) · [当前报告与组件图](../../{DOC.relative_to("docs").as_posix()}/index.html)。failure_ledger_delta=CPU_durability_{step}_no_new_scientific_verdict。\n'
    paths=[]
    for name in ['index.html','runtime_progress.json','local_delivery_checks.json','task_record.json','candidate_source.zip',receipt]:
        rel=DOC/name;copy(O/name,REPO/rel);paths.append(rel.as_posix())
    for name in ['enrich_runtime_report.py','publish_checkpoint_milestone.py']:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    write(statusfile,current+marker+history);paths.append('docs/RESEARCH_STATUS.md')
    write(card,oldcard+entry);paths.append('docs/research_failures/entries/V82-F06.md')
    (W/f'publish_checkpoint_{step:06d}_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(paths),step=step,iteration=s['iteration'],quality_accepted=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);a=p.parse_args();publish(a.step)
