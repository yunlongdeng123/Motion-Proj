"""CPU结果记到原索引与失败卡；不启动GPU或微调。"""
from common import *


def main():
    m=read(O/'manifest.json');check=read(O/'cpu_check.json');assert check['CPU_ready']
    visual=read(O/'input_visual_review.json');assert set(visual['cases'])=={c['case_id'] for c in m['cases']}
    assert not list((O/'inputs').glob('*/result.json')) and not list((O/'inputs').glob('*/mask_result.json'))
    E=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r50';E.mkdir(parents=True,exist_ok=True)
    selection={'task_id':TASK,'run_id':'r50','scene_count':20,'case_count':m['case_count'],
        'case_role':'nuScenes train real DELETE development; no current training',
        'reserve_scenes':m['reserve_scenes'],'base':m['base'],'seed':42,'steps':25,
        'eligible_cases':m['eligible_cases'],'eligible_scenes':m['eligible_scenes'],
        'input_excluded_cases':m['input_excluded_cases'],
        'cases':[{k:c[k] for k in ['case_id','scene','camera','instance_token','source_manifest','source_slice',
                    'median_width','median_height','behind_vehicle_proxy','input_difficulty','difficulty_factors',
                    'input_visual_review','structural_audit_eligible','B_evidence_status']} for c in m['cases']],
        'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'none; model outputs not generated','human_verdict':None}
    dump(E/'selection.json',selection);dump(E/'cpu_check.json',check);dump(E/'input_visual_review.json',visual)
    state=read(O/'controller_state.json');state.update(stage='CPU_ready_waiting_user_GPU',GPU_jobs=0,
        mask_cases=0,windows=0,training_steps=0,eligible_cases=m['eligible_cases'],eligible_scenes=m['eligible_scenes'],
        input_review='main_agent_single_frame_f05; no independent or human verdict')
    dump(O/'controller_state.json',state);dump(E/'controller_state.json',state)
    (REPO/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r50`：按用户最新要求先扩大真实DELETE结构排查，再根据失败造数据和验证内部空间层。20个官方nuScenes train开发场景、49个独立单车目标已完成CPU准备与原视频审核页；抽f05后6例树木/护栏遮挡、逆光或暗处输入暂排除，43例/18景待GPU。另留5景不看RGB、不训练。复用缓存，非700景均匀总体样本。34例有后车投影代理，不代表后车外观证据已确认充分。

统一基线 r46 官方 DriveEditor 原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、微调、时间模块修改或累计多车删除。官方SDK关键帧/插值规则检查与49段原视频实际解码通过；主agent只抽f05核对原目标，SAM身份要开卡后单独查。人工verdict始终空。

当前无GPU，SAM2/DriveEditor均0任务、训练0步；CPU完成后已停在等用户开卡。开卡后先完整SAM+实例检查，再每个合格目标固定一次DELETE。保存原生与写回，粗分薄膜、后车变形、车身向道路延伸；不拿失败生成当训练Y、不逐例调seed或洞规则、不自动微调。r49未见稳定收益的边界及旧产物保留，不恢复其训练。

本地审核 `outputs/v77-real-delete-r50/index.html`；[组件图、输入与停止规则](v77/REAL_DELETE_STRUCTURE_R50.md)，同一失败卡[V77-F02](research_failures/entries/V77-F02.md)。数据盘约余91GiB，无清理或电源操作。GPU批次预计约1–2小时，开卡后按实际首批耗时更新。
''',encoding='utf-8')
    exp=REPO/'docs/EXPERIMENTS.md';text=exp.read_text(encoding='utf-8')
    if 'REAL_DELETE_STRUCTURE_R50.md' not in text:
        row='| WS-V77-TARGET-PROTECTED-20260929 / r50 | 真实DELETE结构排查；20景49单车CPU完成，43例/18景候选，5景隔离；GPU/训练0 | [报告](v77/REAL_DELETE_STRUCTURE_R50.md) |\n'
        text=text.replace('|---|---|---|\n','|---|---|---|\n'+row,1)
        exp.write_text(text,encoding='utf-8')
    card=REPO/'docs/research_failures/entries/V77-F02.md';text=card.read_text(encoding='utf-8')
    if '### r50 CPU：扩大真实DELETE结构排查' not in text:
        text+='''

### r50 CPU：扩大真实DELETE结构排查

在r49稀疏软绑定无稳定视觉增益边界上，按用户最新指令不继续训练或改时间层，先回r46官方原权重+r21完整SAM扩大真实任务。20个官方train开发景49个独立目标已冻结，34例有后车投影；目标RGB/SDK时刻和49原视频核验。主agent抽f05另排除6例遮挡/逆光/过暗输入，43例/18景进GPU候选；不伪称全49例well-observed，后车可见性仍须实际图像确认。缓存池受旧采样影响，不当700景总体评测；另5景排除全部既有来源且不看RGB，后续收益独立检查。

GPU当前无卡、生成与训练均0，因此没有新增模型失败判决。GPU前先核对SAM身份；框交叠不直接认定串实例。真实隐藏区没有GT，失败输出仅定位，未来监督仍是真实视频Y。输入脚手架的曝光间隔门槛、JSON整数及FFmpeg PATH错误已在GPU前修正并保留前版，不归入模型失败。[报告与组件图](../../v77/REAL_DELETE_STRUCTURE_R50.md)。failure_ledger_delta: none（准备记录，无新科学失败）；不新增ID。
'''
        card.write_text(text,encoding='utf-8')
    print('RECORDED_CPU',m['case_count'],flush=True)


if __name__=='__main__':main()
