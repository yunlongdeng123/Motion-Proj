"""收口本轮有界实验；图像评审来自指定subagent，不新增GPU任务。"""
from common import *
from review import review_sentence
import shutil, subprocess


def main():
    controller=read(O/'controller_state.json')
    assert controller['stage']=='complete_pending_human_review' and controller['GPU_jobs']==0
    assert controller['training_steps']==128
    assistant=read(O/'assistant_image_review.json')
    assert assistant['human_verdict'] is None
    assert {r['case_id'] for r in assistant['cases']}==set(EVAL_128)
    summary=read(O/'summary.json')
    evidence=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r48'
    overall=review_sentence(assistant['overall'],'overall')
    hypothesis=review_sentence(assistant['next_module_hypothesis'],'hypothesis')
    rows='\n'.join('| '+r['case_id']+' | '+review_sentence(r['visual_comparison'],'comparison').replace('|','／')+' | '+
        review_sentence(r['regression_risk'],'risk').replace('|','／')+' |' for r in assistant['cases'])
    report=REPO/'docs/v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md'
    text=report.read_text(encoding='utf-8')
    text=text.replace('当前仅CPU准备；训练0步、模型采样0窗，等待用户开GPU。',
        '本轮GPU短循环已完成：128步、20个新采样窗。独立图像review已完成，人工verdict待用户；没有追加GPU任务。')
    text=text.replace('## GPU固定短循环（尚未执行）','## GPU固定短循环（已执行；下列是预先冻结方案）')
    marker='## GPU收口与独立图像review'
    text=text.split('\n'+marker)[0]
    final=f'''
{marker}

按用户最新要求，GPT-5.6 Sol、xhigh（未启用fast）直接检查七例对照图像；GT洞内MAE只归档为辅助记录，不用于视觉质量排序。固定f05的结论如下，抽帧不等于整段视频通过率或时序 verdict。

| case | 视觉比较 | 回退风险 |
|---|---|---|
{rows}

独立评审总结：{overall}

### 模块证据与边界

两例可靠参考的冻结VAE源图与重构图中，保护车大体车身和可见前／后脸仍可识别；不能据此保证细部纹理、视角覆盖或身份绑定充分。所有新窗口实际25次UC与25次C调用通过，C消融任务指令相同，UC新指令另行清空；主干/3D/VAE保持冻结、训练梯度有限。没有发现本轮输入、加载或数值错误。

相同r47权重和指令下，替换错配RGB造成的洞内原生输出差异（0–1，全部10帧）A034=0.000402、A061=0.001143；移除RGB分别0.006600、0.010694。RGB head确实响应不同输入，但具体参考内容对输出的影响较弱。像素差和RMS只是模块敏感性，不是保护车质量或“模型忽略参考”的证明。

代码中六张参考被压成各8×8 token，只有slot级pose；query来自池化UNet激活和时间，BEV/2D几何走并行残差，没有显式把某张crop的某个车身位置绑定到查询中对应保护车区域。这是下一步可以单独验证的融合限制，并非已确定的全部根因。四训练例均有保护车appearance参考；不能把本轮解释成“训练完全没有车身参考”。少量数据、视角差、覆盖和原监督仍未排除。

下一模块假设（独立评审）：{hypothesis}

后续只考虑一项RGB参考融合改动：给参考token与query建立显式空间对应及局部保护区域门控，其他数据、主干、BEV、mask、预算固定；先做正确／错配／无RGB图像对照。O仍是proxy，U仍未知，不能把门控外区域强认作背景，也不能凭单帧确诊删除车特征。本轮没有实现或启动下一实验。M003/M006/A022只用于验证，不能转作训练监督。默认保留r46，不追加步数，不把r48替换成默认权重。

### 执行与资源

从r47 branch_0320开始，389856参数小分支仅4既有训练例，64步后5窗验证完成，再续到128步并验证7窗；另2例各4组模块探针。共20新窗、200原生与200写回PNG；7个独立QUERY，其中5真实DEV、2合成GT DEV，不能把20窗当20个独立case。真实DEV未进入训练；本轮没有未曝光最终测试。

3090 GPU阶段约33分钟（报告开始前）；64/128段训练分别约209/206秒，训练峰值allocated 10.42GiB，推理观测设备使用峰值约23.4GiB。报告82条视频中42条来自原r46/r47，40条新视频已实际解码400帧；本地链接验证完成。磁盘约余92GB。控制器已结束、GPU无计算进程，不自动重启或新增定时任务。

合成GT误差和C/UC记录保存在[summary.json](../autoresearch/worldsim_v77/target_protected_20260929/r48/summary.json)；独立视觉证据见[指定模型review](../autoresearch/worldsim_v77/target_protected_20260929/r48/assistant_image_review.json)。人工0/1/2及视频时序判定保留null。原资产、源工作簿、全部输入、16步断点、64/128权重、优化器和随机状态继续保留在run。

failure_ledger_delta: updated V77-F02（修正消融、参考融合敏感性及有界视觉结果）；不新增failure ID，不认定整体架构或数据分布已被排除。
'''
    report.write_text(text+'\n'+final,encoding='utf-8')
    status=f'''# 当前研究状态

更新：2026-10-07。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

唯一run `WS-V77-TARGET-PROTECTED-20260929/r48` 已完成128步短循环：8个模块探针窗、64步5个验证窗、128步7个验证窗；GPU计算已结束，控制器退出，没有新增任务或定时任务。磁盘约余92GB，无需清理。

用户指定GPT-5.6 Sol、xhigh、不开fast独立看七例图像，GT像素误差仅辅助归档。评审结论：{overall} 单帧不代表整段时序通过；人工0/1/2仍由用户填写。默认继续r46官方原权重+r21完整SAM，不自动推广r48。

旧r47全未知C同时清空任务指令的混杂已修正。新C各组同指令、UC单独清空；原始输入/权重和全部历史结果保留。可靠参考VAE大体结构保留，RGB分支接收与模块响应正常，但正确／错配参考的输出差异弱于有／无RGB；不能把响应或误差当语义收益。

下一项仅聚焦RGB参考到查询的空间融合绑定，数据、主干、BEV、mask与预算不同时改；这是待验证假设，不是已确定的模型架构根因。四训练例已有保护参考；视角、信息覆盖、监督和小预算仍未排除。本轮停止，不自动追加训练或从旧关机记录触发电源动作。

三列视频、原生/写回、独立图像review、参考VAE与模块记录已进入本地 `outputs/v77-priors-r48/index.html`；配置、组件图和完整边界见[r48报告](v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md)。同一失败卡[V77-F02](research_failures/entries/V77-F02.md)已更新。
'''
    (REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
    index=REPO/'docs/EXPERIMENTS.md';text=index.read_text(encoding='utf-8')
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if '| WS-V77-TARGET-PROTECTED-20260929 / r48 |' in line:
            lines[i]='| WS-V77-TARGET-PROTECTED-20260929 / r48 | 同指令消融；128步/20新窗；指定5.6 Sol独立图像review，保留r46；人工待评 | [报告](v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md) |'
    index.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    failure=REPO/'docs/research_failures/entries/V77-F02.md';text=failure.read_text(encoding='utf-8')
    marker='### r48 GPU：参考响应与独立视觉收口'
    text=text.split('\n'+marker)[0]
    text+=f'''

{marker}

固定4train、389856参数分支从r47继续64/128步，主干冻结；8模块探针＋5/7验证共20新窗全部完成，C同指令/UC单独清空、梯度和输出有限、CFG实际调用及写回通过。可靠参考VAE大体车身保留；正确／错配RGB差异A034/A061为0.000402/0.001143，小于移除RGB的0.006600/0.010694，这是敏感性证据而非任务通过率。

按用户要求指定GPT-5.6 Sol/xhigh、不开fast直接审七例图像，结论：{overall} 图像只约束抽取帧，不代填人工0/1/2或视频时序；GT误差不作为质量排序。保留r46、旧r47与所有反例，不推广r48。

参考token仅slot级pose、query缺少显式车身位置对应是下一RGB融合模块假设；四train已有保护车参考，不能归因于完全缺该类数据。数据覆盖、视角/监督与短预算仍开放，不否定完整架构，也不追加训练。控制器/GPU已空，资产、64/128权重和状态保留。[图像/模块证据与组件图](../../v77/TARGET_PROTECTED_SHORT_CYCLE_R48.md)。failure_ledger_delta: updated V77-F02。
'''
    failure.write_text(text,encoding='utf-8')
    for source,name in [(O/'summary.json','summary.json'),(O/'assistant_image_review.json','assistant_image_review.json'),
        (O/'controller_state.json','controller_state.json'),(O/'report_state.json','report_state.json'),
        (O/'training/config.json','training_config.json'),(O/'training/backward_probe.json','backward_probe.json'),
        (O/'training/state.json','training_state.json'),(O/'local_delivery_check.json','local_delivery_check.json')]:
        dump(evidence/name,read(source))
    dump(evidence/'progress.json',{'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r48','parent_run':'r47',
        'stage':'complete_pending_human_review','GPU_jobs':0,'training_steps':128,
        'new_inference_windows':20,'unique_query_cases':7,'real_query_cases':5,'synthetic_query_cases':2,
        'train_cases':TRAIN_IDS,'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02',
        'assistant_review_model':'gpt-5.6-sol','assistant_review_reasoning':'xhigh','fast_enabled':False,
        'human_verdict':None,'temporal_verdict':None,'inputs_preserved':True,'no_auto_extra_training':True})
    subprocess.run([sys.executable,str(REPO/'scripts/build_research_failure_index.py')],cwd=REPO,check=True)
    print('GPU_CLOSEOUT_RECORDED',flush=True)


if __name__=='__main__':main()
