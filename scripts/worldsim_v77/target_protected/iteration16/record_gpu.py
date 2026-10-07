"""同一r49的GPU收口与轻量证据；没有训练、采样或电源操作。"""
from common import *
import argparse, datetime, shutil, subprocess


def sentence(value):
    if isinstance(value, str):return value
    if isinstance(value, dict):return value.get('overall', json.dumps(value, ensure_ascii=False))
    return json.dumps(value, ensure_ascii=False)


def main(mode):
    execution = read(O/'gpu_execution_check.json')
    reviews = {stage: read(O/f'{stage}_assistant_review.json') for stage in ('zero','step64')}
    state = read(O/'controller_state.json')
    assert execution['new_windows']==20 and execution['steps']==64
    assert all(len(review['cases'])==5 for review in reviews.values())
    assert state['phases_complete']==['zero','train64','evaluate64'] and state['GPU_jobs']==0
    proc = subprocess.run(['pgrep','-af','[r]un_cycle.py|[g]pu_experiment.py'],capture_output=True,text=True)
    assert not proc.stdout.strip(), proc.stdout
    state.update(stage='bounded_complete_pending_human_review', training_steps=64,
                 new_inference_windows=20, GPU_jobs=0, human_verdict=None, no_auto_extra_training=True)
    dump(O/'controller_state.json',state)
    last = max(p.stat().st_mtime for p in (O/'evaluation').glob('*/*/*/result.json'))
    summary = {'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r49',
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02; no new ID',
        'training_steps':64,'new_inference_windows':20,'new_result_videos':40,'GPU_jobs':0,
        'last_GPU_window_UTC':datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),
        'GPU_compute_seconds':execution['inference_compute_seconds']+execution['training_compute_seconds'],
        'training_peak_allocated_GiB':execution['training_peak_allocated_GiB'],
        'outcome':'本轮未观察到相对r47/r48_64稳定的真实DELETE视觉升级；薄膜仍在，参考与路由特异性未成立。',
        'default':'r46 official original weights + r21 complete SAM', 'promote_to_default':False,
        'assistant_reviewer':{'model':'gpt-5.6-sol','reasoning_effort':'xhigh','fast_requested':False,
                              'scope':'five cases, f00/f05/f09, direct image review'},
        'human_verdict':None,'temporal_verdict':None,'automatic_extra_training':False,
        'power_action':'none; user notified GPU work finished; historical shutdown authorization not reused'}
    dump(O/'gpu_closeout.json',summary)
    if mode=='prepare':
        print('GPU_FINISHED_CPU_DELIVERY_PENDING',flush=True);return
    delivery=read(O/'local_delivery_check.json')
    assert delivery['new_GPU_windows']==20 and delivery['actual_videos_decoded']==75
    summary['local_delivery']=delivery;dump(O/'gpu_closeout.json',summary)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=Path('/root/autodl-tmp/backups')/('v77_r49_gpu_record_'+stamp)
    for relative in ['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md',
                     'docs/research_failures/entries/V77-F02.md','docs/v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md']:
        dst=backup/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/relative,dst)
    report=REPO/'docs/v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md';text=report.read_text()
    text=text.replace('CPU准备已完成；训练0步、新推理0窗；没有后台等卡进程或定时任务，等待用户开启GPU。',
                      '2026-10-08 GPU收口完成：64步、20新窗口。GPU和控制器已空，用户已获通知可切CPU；没有定时任务。')
    text=text.replace('当前稀疏N没有形成纯背景token/查询格，因此本轮首先检验局部车辆绑定，不能声称已实现整洞背景控制。',
        '实际查账：A061_w08有2个、M003有1个N源patch，其他七例为0；九例f05洞内粗query的N均为0，其他时刻仅有零星N。不能把“f05没有N对应”概括为全程无N源token，本轮也没有实现整洞背景控制。')
    text=text.replace('A034只有第一层粗格的局部绑定，后续10×18/5×9多数仍U；A061也有未覆盖区域。',
        'A034在18×32粗格仅有局部绑定，CPU/训练较低尺度10×18/5×9多数仍U；A061也有未覆盖区域。GPU推理的实际尺寸在后文单独记录。')
    text=text.replace('r49效果栏仍为空，人工verdict未填写。','这描述CPU交付时刻；现已加入GPU视频与直接图像对照，人工verdict仍未填写。')
    text=text.replace('## GPU短循环（尚未运行）','## 已执行的GPU短循环')
    text=text.replace('零训练入口（用户开GPU后手动运行）：','手动零训练入口（本轮已运行）：')
    text=text.replace('CPU阶段没有启动控制器。','CPU准备阶段没有启动控制器；本轮用户开卡后才手动启动。')
    text=text.replace('failure_ledger_delta: updated V77-F02（r49融合入口准备与有限对应覆盖）；模型效果未运行，不新增failure ID。',
                      'failure_ledger_delta: updated V77-F02（CPU覆盖边界及GPU未观察到稳定视觉升级）；不新增failure ID。')
    table=[]
    for row in reviews['step64']['cases']:
        table.append(f'| {row["case_id"]} | {sentence(row["ranking"]).replace("|","/")} | {sentence(row["film"]).replace("|","/")} |')
    section='''## GPU结果与证据边界

零训练9窗未达标后，主agent看完五例且指定GPT-5.6 Sol/xhigh、不开fast独立看固定三帧，才登记唯一64步准入。64步后十一窗全部完成，包括两个主要真实例的同权重routing_off。没有延长到128、扩数据、换mask、改seed或加入去噪抑制。

| case | 64步后的独立助手图像判断 | 额外轮廓 / 薄膜 |
|---|---|---|
'''+ '\n'.join(table)+f'''

64步耗时 {execution['training_compute_seconds']:.1f}s，峰值分配显存 {execution['training_peak_allocated_GiB']:.2f}GiB；20窗采样耗时合计 {execution['inference_compute_seconds']:.1f}s，以上不含模型加载和CPU评审。训练范围共389856个标量参数，其中60个参数张量相对初始权重发生变化；主干/3D/VAE保持冻结，每步loss/梯度有限，实际case顺序与RGB/geometry dropout逐步匹配r48_64。400张原生/写回PNG实际解码，CFG为25C+25UC、C同指令、UC单独清空、洞外保留和alpha=1区等于原生均通过。这些是工程与刺激有效性证据，不是质量分。

实际576×1024推理的四个注入block为18×32、18×32、18×32、9×16；10×18/5×9是CPU代表尺度及320×576训练尺度，不应当成全部GPU推理尺寸。偏置在有证据的block实际生效，未知或无RGB时中性；日志中的biased_pairs是候选logit数量，不是成功注意力质量。

两个主要真实例中correct/wrong/no_RGB及同权重routing_off目视近同，没有观察到参考或路由的独立稳定增益；薄膜在原生已有，不能整体归因于固定alpha写回。A022仍有比r46明显的宽暗路面补丁，但没有r48_128的大块结构崩坏；它没有可绑定patch，因此不能把零训练的暗影直接归因于空间偏置。两个合成例保住相对原模型的恢复收益，但没有证明本轮升级。

**本轮不推广新权重，默认仍r46。** 图像判断只覆盖五例各f00/f05/f09，真实隐藏区无GT；人工0/1/2与视频时序verdict留空。不能将当前稀疏、无参数软偏置的失败外推为完整空间绑定假设、全部架构或数据分布的否定。

下一项候选先检验“已知B查询只选对应B的可靠源patch与null”的更明确绑定，未知查询继续走原路径；未知源不能被误标背景。本轮软偏置对未知源保持0，混合/未知来源仍可能竞争，这是一项接口限制，不是已经测得的注意力因果。不要靠增加参考数量或延长训练规避这一项检验；主干去噪抑制仍作为后续独立改动，不与绑定同时添加。

本地审核交付9卡、40个新结果视频与35个历史视频；75个视频实际解码为1024×576、10帧/1秒，图片链接与JS通过。HTML保留零训练、64步、原生/固定写回、参考控制、关闭路由、CPU几何图和直接三帧图板。全部旧资产、权重和反例保留，数据盘约余90.8GiB，无需清理。

轻量GPU证据见同run的[gpu_execution_check.json](../autoresearch/worldsim_v77/target_protected_20260929/r49/gpu_execution_check.json)、[独立64步图像review](../autoresearch/worldsim_v77/target_protected_20260929/r49/step64_assistant_review.json)、[实际尺度/来源](../autoresearch/worldsim_v77/target_protected_20260929/r49/actual_eval_coverage.json)、[交付检查](../autoresearch/worldsim_v77/target_protected_20260929/r49/local_delivery_check.json)。模型权重、原始PNG和完整视频只留数据盘和本地输出，不复制进Git。
'''
    marker='## GPU结果与证据边界';text=text.split(marker)[0].rstrip()+'\n\n'+section
    report.write_text(text,encoding='utf-8')
    status='''# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r49` 已完成64步、20新窗口及两阶段五例三帧独立助手review。GPU和控制器已空，没有自动追加训练或定时任务；已通知用户可切CPU。本次没有电源操作。

没有观察到相对r47/r48_64稳定的真实DELETE视觉升级：A034/A061薄膜与额外轮廓仍在，correct/wrong/no_RGB与同权重routing_off目视近同；A022仍逊于r46，两个合成例保住旧收益。工程检查、偏置生效、主干冻结和同预算输入顺序通过，不等于视觉收益；人工verdict与整段时序未判。

默认仍r46官方原权重+r21完整SAM，不推广r49。空间对应稀疏，f05洞内无N；更正来源概括：A061有2、M003有1个N源patch，非全程全无N。下一项先检验已知B查询更明确选择对应B来源、未知保持原路径；不追加步数，不与主干去噪控制同时修改。完整组件图与结果见[r49报告](v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。

本地审核`outputs/v77-priors-r49/index.html`含9卡、75视频并实际解码；原生/写回、参考/路由控制、三帧图板均保留。数据盘约余90.8GiB，原权重/输入/断点均在，无清理需求。
'''
    (REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
    index=REPO/'docs/EXPERIMENTS.md';lines=index.read_text().splitlines()
    row='| WS-V77-TARGET-PROTECTED-20260929 / r49 | 实例/位置软绑定；64步/20新窗；两阶段5.6 Sol直接看图未见稳定升级；保留r46，GPU已空 | [报告](v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md) |'
    assert sum('/ r49 |' in line for line in lines)==1
    index.write_text('\n'.join(row if '/ r49 |' in line else line for line in lines)+'\n',encoding='utf-8')
    card=REPO/'docs/research_failures/entries/V77-F02.md';text=card.read_text()
    text=text.replace('稀疏N未形成纯背景patch，M013没有可靠参考patch。',
        '准备阶段对N概括过度（GPU查账：A061有2、M003有1个N源patch，f05洞内仍无N）；M013没有可靠参考patch。')
    marker='### r49 GPU：软绑定无稳定视觉增益'
    failure='''### r49 GPU：软绑定无稳定视觉增益

零训练9窗直接看图未达标后，仅一次64步/11窗；合计20新窗与400张原生/写回PNG工程检查通过，主干冻结、389856参数范围、同C指令/UC单独清空和与r48_64同数据顺序/dropout通过。有证据的query实际加了偏置，因此不是入口未启用；GT Y未进条件、真实DEV不训练。

两阶段指定GPT-5.6 Sol/xhigh、不开fast直接看五例三帧：A034/A061与r47/r48大体同档，薄膜仍在，正确/错配/无RGB及同权重关闭路由没有独立可见增益；A022仍有比r46明显的暗补丁，M003/M006保住旧合成收益。原生已有伪影，不能整体归因alpha；没有续128，不推广r49，默认r46保留。人工与视频时序未判。

本轮仅验证稀疏、保守proxy上的软偏置，不否定完整绑定或整体架构。实际推理尺寸为18×32三处+9×16；A061有2、M003有1个N源patch，各例f05洞内仍无N，修正CPU报告“全无N源patch”的过度概括。已知B查询仍允许未知来源中性竞争是下一接口候选，并非已证实唯一因果。见[组件图、直接图像与执行记录](../../v77/TARGET_PROTECTED_SPATIAL_BINDING_R49.md)。failure_ledger_delta: updated V77-F02；无新增ID。
'''
    text=text.split(marker)[0].rstrip()+'\n\n'+failure;card.write_text(text,encoding='utf-8')
    evidence=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r49'
    for name in ['controller_state.json','review_check.json','zero_assistant_review.json','step64_assistant_review.json',
                 'zero_shot_gate.json','zero_execution_check.json','gpu_execution_check.json',
                 'actual_eval_coverage.json','gpu_closeout.json','local_delivery_check.json']:
        dump(evidence/name,read(O/name))
    subprocess.run([sys.executable,str(REPO/'scripts/build_research_failure_index.py')],cwd=REPO,check=True)
    print('R49_GPU_RECORDED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','record'])
    main(p.parse_args().mode)
