"""精确登记三组已授权消融及真实准入/队列，保留旧实验资产。"""
from pathlib import Path
import json,shutil
W=Path(__file__).resolve().parent;BASE=W.parent.parent
O=BASE/'outputs/gn11_stage1_v2_20261019'
REPO=W.parent/'r15_closure_run/publish_repo'
DOC=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019')
CODE=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v2_20261019')
def long(p):return Path(chr(92)*2+'?'+chr(92)+str(p.resolve()))
def write(p,t):long(p).parent.mkdir(parents=True,exist_ok=True);long(p).write_text(t,encoding='utf-8')
def copy(a,b):long(b).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(long(a),long(b))

def main():
    r=json.loads((W/'ablation_runtime.json').read_text(encoding='utf-8'))
    registry=r['registry'];real=r['arms']['real_only']
    if not real['gate'] or not real['status']:raise RuntimeError('real gate and actual training required')
    if real['gate']['optimizer_updates']!=0:raise RuntimeError('gate must be zero update')
    heading='## 2026-10-19：用户新增三组30K来源消融，最多两路并发'
    card=REPO/'docs/research_failures/entries/V82-F06.md';old=card.read_text(encoding='utf-8')
    if heading in old:raise RuntimeError('registration already published; use actual later milestone')
    row='| WS-V82-R15-GN11-STAGE1-V2-ROAD-ABLATIONS-20261019 / real_only_seed0、prior_only_seed0、ownership_only_seed0 | 三组fresh第一阶段30K：仅跨帧真实观测、仅供体先验、仅归属修复；相同五身份49帧/seed0，最多两路并发 | [四组组件与实际队列](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/gn11_stage1_v2_20261019/ablations/index.html) |'
    experiments=REPO/'docs/EXPERIMENTS.md';index=experiments.read_text(encoding='utf-8')
    if 'WS-V82-R15-GN11-STAGE1-V2-ROAD-ABLATIONS-20261019' in index:raise RuntimeError('duplicate experiment group registration')
    lines=index.splitlines();anchor=next(i for i,line in enumerate(lines) if line.startswith('| WS-V82-R15-GN11-STAGE1-V2-20261019 /'))
    lines.insert(anchor,row)
    record=json.loads((O/'task_record.json').read_text(encoding='utf-8'))
    record.update(state='FRESH_30K_MAIN_WITH_THREE_AUTHORIZED_ABLATIONS',
        ablation_group_task='WS-V82-R15-GN11-STAGE1-V2-ROAD-ABLATIONS-20261019',
        additional_authorized_fresh_updates=90000,max_concurrent_training_including_main=2,
        ablation_controller_pid=registry['controller_pid'],ablation_runs={})
    for name,a in r['arms'].items():
        record['ablation_runs'][name]=dict(updates_authorized=30000,seed=0,second_stage_updates=0,
            status=(a['status'] or {}).get('status','queued'),updates_executed=(a['status'] or {}).get('iteration',0),
            actual_optimizer_step_counts=(a['status'] or {}).get('optimizer_step_counts'),
            actual_zero_update_gate_passed=bool(a['gate']),road_weights=a['manifest']['road_weights'])
    write(O/'task_record.json',json.dumps(record,indent=2,ensure_ascii=False))
    details=f'''用户最新明确新增三组fresh30K消融：real_only（归属修复＋同世界位置跨帧真实观测，先验目标权重0）、prior_only（归属修复＋左右供体先验，真实目标权重0）、ownership_only（仅归属修复，两种道路目标权重均0）。三组同五身份/49帧/原始输入初始化协议/seed0，各30000步、第二阶段0。静态表示、监督路由、体积排除与动态完整mask alpha MSE100保持相同，冻结query/provenance不改；只在原OBSERVED/PRIOR各自区域启用相应来源，UNKNOWN无道路目标，不扩供体。对照按相同步数和真实曝光/先验/未知区域分别复核；所有49帧属于训练重建，3D框为额外约束输入。仅归属组与原A原生基线比较修复组合效果，模块内部单项尚未拆分，旧初始GPU张量快照不可得。

2026-10-19T09:58:33UTC已启动新增调度器PID{registry['controller_pid']}，最多两路训练（包含主实验和gate）。当前采样{r['collected_utc']}：主PID2145为{r['main']['iteration']}/30000，real_only PID{real['status']['pid']}为{real['status']['iteration']}/30000，真实static/dynamic优化器计数与各自进度一致。real_only已实际CUDA零更新准入：时间-0.48/0.48静态固定相机RGB差0，梯度组6/9/1有限、三个优化器计数0，真实来源权重1/.1、先验权重0。前100选帧/时间扰动和全部优化器LR与当前完整版实际日志逐项相同。prior_only、ownership_only仍排队，未伪报准入或训练。每个新worker Torch分配上限10GiB，现有主训练源码、目标、输入、预算均未修改。

共享准入只允许当前主训练与controller登记子进程，各组独立目录、独立锁、独立gate与六个5K保存节点；不自动重启或重放。GPU本轮：{r['gpu']}。磁盘空闲{r['disk_free_bytes']/1e9:.3f}GB，保留全部剩余节点估需求{r['disk_required_estimate_bytes']/1e9:.3f}GB；该配置点数加10%/增密结束后实际点数估算含8GB共享余量，不是严格点数上界。保存不足时等待并保留内存，历史输入/checkpoint不删。全部四组训练、节点核验、零更新49、下载、完整画面对照与Git交付完成前不关机，preflight包括新增controller及所有在训/排队工作。human_verdict=null，效果尚无结论。'''
    paths=[]
    for p in sorted((O/'ablations').iterdir()):
        if p.is_file():rel=DOC/'ablations'/p.name;copy(p,REPO/rel);paths.append(rel.as_posix())
    for name in ['index.html','candidate_source.zip','local_delivery_checks.json','runtime_progress.json','task_record.json']:
        rel=DOC/name;copy(O/name,REPO/rel);paths.append(rel.as_posix())
    for p in sorted(O.glob('checkpoint_*_audit.json')):
        rel=DOC/p.name;copy(p,REPO/rel);paths.append(rel.as_posix())
    names=['build_road_ablations.py','ablation_support.py','run_road_ablation_group.py','deploy_ablations.py',
           'collect_ablations.py','check_ablation_contracts.py','build_ablation_report.py','publish_ablation_registration.py',
           'enrich_runtime_report.py']
    for name in names:
        rel=CODE/name;copy(W/name,REPO/rel);paths.append(rel.as_posix())
    for p in sorted((W/'ablations').rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.json','.md']:
            relative=p.relative_to(W/'ablations')
            if len(relative.parts)>2 and relative.parts[1]=='candidate':
                shared=W/'candidate'/Path(*relative.parts[2:])
                if shared.is_file() and p.read_bytes()==shared.read_bytes():continue
            rel=CODE/'ablations'/p.relative_to(W/'ablations');copy(p,REPO/rel);paths.append(rel.as_posix())
    statusfile=REPO/'docs/RESEARCH_STATUS.md';status=statusfile.read_text(encoding='utf-8')
    marker='## 已完成的原A/B基线与历史证据'
    current,history=status.split(marker,1)
    current+=f'新增授权与队列：用户已授权real_only（仅跨帧真实道路）、prior_only（仅左右供体先验）、ownership_only（仅归属修复）各fresh30K，第二阶段0，保持同五身份/49帧/seed0，三组归属修复与完整版相同、仅道路来源权重切换。当前controller PID{registry["controller_pid"]}最多两路并发，real_only PID{real["status"]["pid"]}已{real["status"]["iteration"]}步并通过实际CUDA零更新准入，另外两组排队。源码/输入未改主训练，禁止重复启动或自动重放。全部主实验和三组消融的训练、评估、下载、完整画面对照与Git完成后再关机，preflight覆盖新增controller与所有在训/待执行worker。\n\n[三组30K消融与四组对照合同]('+str((DOC/'ablations/index.html').relative_to('docs')).replace('\\','/')+')。来源梯度、GPU准入和资源证据见报告及既有V82-F06追加；效果尚未验收。\n\n'
    write(statusfile,current+marker+history);paths.append('docs/RESEARCH_STATUS.md')
    write(card,old+'\n\n'+heading+'\n\n'+details+'\n\n[实际来源消融、组件图和队列](../../'+str((DOC/'ablations/index.html').relative_to('docs')).replace('\\','/')+')。failure_ledger_delta=authorized_source_factorial_ablation_admission_no_quality_verdict。\n')
    paths.append('docs/research_failures/entries/V82-F06.md')
    write(experiments,'\n'.join(lines)+'\n');paths.append('docs/EXPERIMENTS.md')
    paths=list(dict.fromkeys(paths))
    (W/'publish_ablation_paths.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(paths),real_only_step=real['status']['iteration'],max_concurrent_training=2)))

if __name__=='__main__':main()
