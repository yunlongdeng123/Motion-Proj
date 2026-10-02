"""r44工程前置关系检查收口；保留r43失败像素及旧合法正控制。"""
from pathlib import Path
import json,shutil,subprocess,html
P=Path('/root/autodl-tmp/motion_proj_v77')
T=Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')


def main():
    r=json.loads((T/'r44/replay.json').read_text());backup=T/'r44/before_changes';backup.mkdir(exist_ok=True)
    files=['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md','docs/v77/TARGET_PROTECTED_SPACE_FIRST_R42_R43.md']
    for rel in files:
        out=backup/(Path(rel).name+'.before_relation_record')
        if not out.exists():shutil.copy2(P/rel,out)
    section='''## r44：把明确的保护车关系放在SAM之前

工程回归固定r43全部4轨迹与旧合法Q060，不新增来源、位置或速度。生成器必须先声明主B实例；只用GT相机／位姿与最终H检查至少3帧存在A整体在B前且H可能碰到B的机会。此条件沿用已有0.3m严格包络深度规则，框交叠只是可能交叠，不能替代实例轮廓或证明道路合法。所有真实显露、邻车、未知区域及独立QA门槛保持。

```mermaid
flowchart LR
 A[固定A轨迹＋显式B身份] --> P[GT投影与最终H\n必要深度／接触关系]
 P --> Q[SAM实例与显露检查]
 Q --> V[独立输入QA＋合法条件QA]
 V -.合格后.-> T[同预算模型对照]
 Y[真实Y] -.仅监督／离线质检.-> Q
```

G001/G002/G004在昂贵标签队列之前被拒；G003保留待完整检查，但旧完整检查和独立QA已证实缺少互补显露，因此不重跑SAM、不准入。Q060使用既有真实主B，前置检查通过；其主B存于retained_instances，首次回归由于只看source actors列表而报错，补入同一既有GT轨迹后恢复，无像素、位姿或阈值变化，原脚本已备份。

四个小型语义回归测试通过：二维相交而深度反转、灰洞与B分离、错身份／错时刻、明确任务到标签队列的过滤。真实回归150帧用CPU约0.34秒。它只减少无效SAM任务，不产生新的合格数据或模型收益。静止B/慢ego是r43的单一有界过程选择，不是全部Temporal reveal定义；旧Q060的A/B均运动，仍是合法正控制。

下一轮生成器应从明确B和其可显露过程共同选A，而不是仅凭车道能放下，再由后处理发现目标不对。此处不追加同来源位置／速度网格。真实DELETE两侧跨场景收益仍未证明，正式A/B/C仍0步，未满足关机条件。failure_ledger_delta: updated V77-F02。
'''
    report=P/files[-1];text=report.read_text();marker='## r44：把明确的保护车关系放在SAM之前'
    if marker in text:text=text.split(marker)[0].rstrip()+'\n\n'
    report.write_text(text+'\n'+section)
    (P/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

2026-10-02，wm-3090-1001，v77。Temporal reveal＋合法projected actor-state，先验证条件，再考虑surfel。

r42取消每scene前三窗口截断，440→612窗口，但旧过程门仍0来源，控制关闭。r43整体取消16m距离代理后得到6scene、13次空间尝试、4候选。完整实例检查0合格；独立QA G001/G002/G003/G004为0/0/1/0。前两例主B深度顺序不成立，G004洞与来源B分离，G003缺乏同一B互补显露。空间无碰撞不代表遮挡任务正确。全部失败对照保留；此来源对照已结束。

r44补入显式主B与前置必要关系检查，固定四例的CPU回归提前拒绝3例，旧Q060正控制通过。4个语义测试通过。GT包络只排除不可能的既定任务，不证明实际分割、显露或道路合法；G003仍不能准入。已结束CPU/GPU作业，没有新控制器或定时任务。

新数据准入0，50条完整配方未齐，正式A/B/C训练0步，身份分支及surfel未启动。旧Q060输入／条件QA2保留，Q046条件QA1，r36过滤未推广。GT相机／轨迹／LiDAR为明示POC辅助；条件只读最终H遮后RGB，Y仅监督／离线QA。人工verdict空。

下一步从明确保护对象与可显露过程共同构造A；保持实际空间和遮后条件质量要求，不追加旧六来源的位置／速度网格，也不把静止B／慢ego这一过程当全部训练配方。先获得跨scene可用条件，再启动同预算模型对照。工程修复不计作真实DELETE增益。

当前本地审核页`outputs/v77-target-protected-r43/index.html`：4例12视频、360帧全部解码，人工评分空，第三列是灰洞输入而非补景输出。报告及组件图：[r42–r44](v77/TARGET_PROTECTED_SPACE_FIRST_R42_R43.md)。旧r38等历史结果保留。

数据盘约44GiB可用，本轮没有删除文件。真实DELETE保护车与无车背景两侧跨scene收益未达到，未满足关机条件。沿用[V77-F02](research_failures/entries/V77-F02.md)，当前唯一task为WS-V77-TARGET-PROTECTED-20260929。
''')
    path=P/'docs/EXPERIMENTS.md';lines=path.read_text().splitlines();key='| WS-V77-TARGET-PROTECTED-20260929 / r44 |'
    line=key+' 显式主B必要关系前置；4固定失败候选提前拦3，Q060保留；4测试通过，新数据／模型收益0 | [记录](autoresearch/worldsim_v77/target_protected_20260929/r44/closeout.json) |'
    if any(x.startswith(key) for x in lines):lines=[line if x.startswith(key) else x for x in lines]
    else:lines.insert(next(i for i,x in enumerate(lines) if x.startswith('|---'))+1,line)
    path.write_text('\n'.join(lines)+'\n')
    path=P/'docs/research_failures/entries/V77-F02.md';text=path.read_text();mark='## r44：显式主B关系前置'
    if mark in text:text=text.split(mark)[0].rstrip()
    path.write_text(text+'\n\n'+mark+'\n\n固定r43四例回归，G001/G002/G004不能建立声明的主B关系，可在SAM前拒绝。G003必要关系通过仍因显露证据不足拒绝。旧Q060真实保护实例来自retained_instances，必须按明确身份取位姿，不可用source第一辆车替代；补齐该metadata读取后正控制通过。4个语义测试通过，CPU约0.34秒；不读取Y、不改H、不运行新GPU或训练。新知识：空间碰撞与身份遮挡任务需分层，必要关系筛选不能替代数据质量。固定来源停止，下一步联合声明主B与显露过程。failure_ledger_delta: updated V77-F02。\n')
    review=T/'r43/review';path=review/'index.html';text=path.read_text()
    start='<!-- primary-relation-r44 -->';end='<!-- /primary-relation-r44 -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    note=start+'<p class="notice"><b>r44 工程补充：</b>新增显式主B的前置关系检查，G001/G002/G004可在SAM之前拒绝；G003仍因缺少其他帧真实显露证据不能准入。旧合格Q060通过正控制。4项测试通过；没有新训练或补景输出。<a href="primary_relation_replay.json">逐帧几何证据</a>（包络可能相交不等于真实mask相交）。</p>'+end
    text=text.replace('<button id="export">',note+'<button id="export">')
    path.write_text(text);shutil.copy2(T/'r44/replay.json',review/'primary_relation_replay.json')
    subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python',str(P/'scripts/build_research_failure_index.py')],cwd=P,check=True)
    print('PRIMARY_RELATION_RECORDED',r['early_rejected'],flush=True)


if __name__=='__main__':main()
