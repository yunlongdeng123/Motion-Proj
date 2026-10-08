"""CPU结果记到原索引与失败卡；不启动GPU或微调。"""
from common import *
from collections import Counter
import re


def main():
    m=read(O/'manifest.json');check=read(O/'cpu_check.json');assert check['CPU_ready']
    visual=read(O/'input_visual_review.json');assert set(visual['cases'])=={c['case_id'] for c in m['cases']}
    quality=read(O/'input_quality_review.json')
    assert check['independent_quality_score2_only'] and m['eligible_scenes']==20
    n=m['case_count'];reviewed=m['input_candidate_count'];rejected=len(m['input_excluded_cases'])
    uncertain=len(m['quality_uncertain_cases']);b=Counter(c['B_evidence_status'] for c in m['cases'])
    assert not list((O/'inputs').glob('*/result.json')) and not list((O/'inputs').glob('*/mask_result.json'))
    E=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r50';E.mkdir(parents=True,exist_ok=True)
    selection={'task_id':TASK,'run_id':'r50','scene_count':20,'case_count':m['case_count'],
        'case_role':'nuScenes train real DELETE development; no current training',
        'reserve_scenes':m['reserve_scenes'],'base':m['base'],'seed':42,'steps':25,
        'eligible_cases':m['eligible_cases'],'eligible_scenes':m['eligible_scenes'],
        'input_excluded_cases':m['input_excluded_cases'],'reviewed_candidates':reviewed,
        'quality_uncertain_cases':m['quality_uncertain_cases'],
        'cases':[{k:c[k] for k in ['case_id','scene','camera','instance_token','source_manifest','source_slice',
                    'median_width','median_height','behind_vehicle_proxy','input_difficulty','difficulty_factors',
                    'input_visual_review','input_quality_score','structural_audit_eligible','B_evidence_status']} for c in m['cases']],
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'none; model outputs not generated','human_verdict':None}
    dump(E/'selection.json',selection);dump(E/'cpu_check.json',check);dump(E/'input_visual_review.json',visual)
    dump(E/'input_quality_review.json',quality)
    archive=read(O/'quality_archive.json')
    dump(E/'quality_exclusions.json',{'policy':archive['policy'],
        'cases':[{k:c[k] for k in ['case_id','scene','camera','instance_token','quality_queue_role','input_quality_review']} for c in archive['cases']]})
    state=read(O/'controller_state.json');state.update(stage='CPU_ready_waiting_user_GPU',GPU_jobs=0,
        mask_cases=0,windows=0,training_steps=0,eligible_cases=m['eligible_cases'],eligible_scenes=m['eligible_scenes'],
        input_review=m['input_quality_scope'])
    dump(O/'controller_state.json',state);dump(E/'controller_state.json',state)
    (REPO/'docs/RESEARCH_STATUS.md').write_text(f'''# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r50`：先扩大真实DELETE单帧结构排查，再按失败造数据。按用户最新要求，subagent已独立检查{reviewed}例f00/f05/f09真实原图与目标crop、可用B参考；0/1分{rejected}例退队列，补齐20个官方train场景、{n}个2分目标，每景2–3个不同实例。uncertain输入{uncertain}例未准入，人工verdict仍空。请求配置gpt-6-sol/xhigh，未启用fast，工具未返回实际模型身份。评分是抽帧输入准入，不代替SAM身份或生成验收。

统一基线r46官方DriveEditor原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、微调、时间模块修改或累计多车删除。官方SDK时刻检查和全部准入原视频实际解码通过。缓存池非700景均匀总体样本。另5景不看RGB、不训练，旧val隔离集不动。后车证据分开登记：{dict(b)}；A合格不保证B隐藏区域证据充分。

当前无GPU，SAM2/DriveEditor均0任务、训练0步；CPU完成后已停在等用户开卡。开卡后先完整SAM+实例检查，再每个合格目标固定一次DELETE。保存原生与写回，粗分薄膜、后车变形、车身向道路延伸；不拿失败生成当训练Y、不逐例调seed或洞规则、不自动微调。r49未见稳定收益的边界及旧产物保留，不恢复其训练。

本地 `outputs/v77-real-delete-r50/index.html` 只显示20景2分队列，退队列原图和理由单独归档。[组件图、输入与停止规则](v77/REAL_DELETE_STRUCTURE_R50.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。数据盘约余91GiB，无清理或电源操作。GPU批次预留约1–2小时，开卡后按实际耗时更新。
''',encoding='utf-8')
    exp=REPO/'docs/EXPERIMENTS.md';text=exp.read_text(encoding='utf-8')
    row=f'| WS-V77-TARGET-PROTECTED-20260929 / r50 | 独立质检{reviewed}例、0/1分{rejected}例剔除；补齐20景{n}个2分单车目标；GPU/训练0 | [报告](v77/REAL_DELETE_STRUCTURE_R50.md) |'
    pattern=r'^\| WS-V77-TARGET-PROTECTED-20260929 / r50 \|.*$'
    text=re.sub(pattern,lambda _:row,text,flags=re.M) if re.search(pattern,text,flags=re.M) else text.replace('|---|---|---|\n','|---|---|---|\n'+row+'\n',1)
    exp.write_text(text,encoding='utf-8')
    card=REPO/'docs/research_failures/entries/V77-F02.md';text=card.read_text(encoding='utf-8')
    if '### r50 CPU：扩大真实DELETE结构排查' not in text:
        text+='''

### r50 CPU：扩大真实DELETE结构排查

在r49稀疏软绑定无稳定视觉增益边界上，按用户最新指令不继续训练或改时间层，先回r46官方原权重+r21完整SAM扩大真实任务。20个官方train开发景49个独立目标已冻结，34例有后车投影；目标RGB/SDK时刻和49原视频核验。主agent抽f05另排除6例遮挡/逆光/过暗输入，43例/18景进GPU候选；不伪称全49例well-observed，后车可见性仍须实际图像确认。缓存池受旧采样影响，不当700景总体评测；另5景排除全部既有来源且不看RGB，后续收益独立检查。

GPU当前无卡、生成与训练均0，因此没有新增模型失败判决。GPU前先核对SAM身份；框交叠不直接认定串实例。真实隐藏区没有GT，失败输出仅定位，未来监督仍是真实视频Y。输入脚手架的曝光间隔门槛、JSON整数及FFmpeg PATH错误已在GPU前修正并保留前版，不归入模型失败。[报告与组件图](../../v77/REAL_DELETE_STRUCTURE_R50.md)。failure_ledger_delta: none（准备记录，无新科学失败）；不新增ID。
'''
        card.write_text(text,encoding='utf-8')
    title='### r50 CPU：独立输入质检准入（替代上一项主agent粗检查）'
    if title not in text:
        text+=f'''\n\n{title}

按用户要求，subagent独立看真实三帧与B参考，首49例仅25例2分、24例1分；低质量输入退队列，补齐20景{n}个2分不同车辆目标。累计检查{reviewed}例，0/1分{rejected}例剔除，uncertain输入{uncertain}例未准入。取代先前43例/18景的主agent粗准入；原图、旧清单和全部独立理由保留。B证据与A质量分开，SAM实例仍待开卡验收。

本次是输入质量调整，GPU/生成/训练均0，尚无新模型失败或方法收益。缓存采样偏差及5景隔离明示；失败输出只能定位，未来Y必须是真实视频。[报告与组件图](../../v77/REAL_DELETE_STRUCTURE_R50.md)。failure_ledger_delta: none；仍更新同卡，不新建ID。
'''
        card.write_text(text,encoding='utf-8')
    print('RECORDED_CPU_QUALITY',n,reviewed,rejected,uncertain,flush=True)


if __name__=='__main__':main()
