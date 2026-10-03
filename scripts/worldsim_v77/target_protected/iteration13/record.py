"""保存本次CPU交付；GPU成绩只有实际运行后才能写。"""
from common import *
import shutil
from collections import Counter

def main():
    pre=read(O/'preflight.json');inv=read(O/'condition_inventory.json');plan=read(O/'manifest.json')
    assert pre['stage']=='CPU_ready_GPU_not_run'
    E=REPO/'docs/autoresearch/worldsim_v77/target_protected_20260929/r45';E.mkdir(parents=True,exist_ok=True)
    for f in ['preflight.json','condition_inventory.json','extraction_plan.json','extract_state.json','dedup.json']:
        shutil.copy2(O/f,E/f)
    dump(E/'run.json',{k:v for k,v in plan.items() if k!='cases'}|{'cases':[{k:c[k] for k in ['case_id','split','scene','type']} for c in plan['cases']],
        'phase':'CPU_ready_waiting_user_GPU','training_steps':0,'new_inference_windows':0,
        'failure_ledger_delta':'V77-F02: solid cuboid projection labels adjacent road; conservative core replaces it before training; no scientific result'})
    rows=inv['cases'];counts=Counter(c['split'] for c in rows)
    table='\n'.join(f"| {c['case_id']} | {c['split']} | {c['scene']} | {c['hole_condition_fraction']['O']:.1%} | {c['hole_condition_fraction']['N']:.2%} | {c['hole_condition_fraction']['U']:.1%} |" for c in rows)
    report=f'''# r45：O/N/U/Q 小分支最小验证

task `WS-V77-TARGET-PROTECTED-20260929/r45`，failure refs `V77-F02`。本次仅完成 CPU 准备；训练 0 步，新推理 0 窗。用户最新要求完成 CPU 后通知开 GPU。

```mermaid
flowchart LR
 G[GT相机／保留车辆框／同窗LiDAR] --> C[O／N／U／Q]
 C --> A[小Adapter · 157888参数]
 X[视频先遮洞再缩放] --> D[原始DriveEditor · 主干冻结]
 A --> D
 Y[真实Y · 仅合成监督] -. loss .-> A
 D --> V[固定真实DELETE · 无条件／全未知／有条件]
```

## 本次范围

复用 {counts['train']} 个旧训练例、{counts['validation']} 个跨训练场景的合成开发例、{counts['real_DEV']} 个固定真实 DELETE 开发例。没有新造遮挡，没有数据工厂搜索，没有新增 SAM/身份系统、encoder、2DGS 或 surfel。两个密集训练例共用一个场景和 A 位姿，但实际 Y/H 不全相等；保留并明示相关性，不虚报独立场景。全部评测例是已曝光 DEV，不是最终泛化测试。

O 是保留车辆 GT 包络的保守内核，Q=0.5；不是由隐藏 Y 推出的精确轮廓。整块框投影的初始检查在 M042 将相邻道路也标成车辆，已保存在 `r45/before_changes/conditions_solid_cuboid`。统一改为离每个实例投影边缘超过短边25%的内核，最小2px；其余边缘为 U，未按 case 调参。此为有噪声的几何先验，不能宣称 O 每像素都是真车。GT 轨迹/相机是明示 POC 辅助。

N 只来自固定窗口内40m范围的实测 LiDAR 背景返回，排除 source H 和所有实体包络（外扩2px），保留每源相机像素最近返回；查询时也排除保留实体。没有实测返回的位置不作 N。它包含道路、护栏、墙等非actor背景，不只限于地面：初始地面拟合筛选使部分有实测背景的例无N，旧图保留在 `r45/before_changes/conditions_ground_only`，因此去掉本任务不需要的地面高度拟合，未填补没有实测依据的区域。Q为启发式可信度，不是校准概率。

14 个缺失的小文件从公共包定向提取，不下载新模型。首轮按RGB线索定位分片有一个LiDAR文件不在04，记录缺项后在已有候选01找到；未重复读取已确认缺失的分片。相机时间没有完整几何支撑时保持未知。条件图不使用 Y 纹理、synthetic actor RGB 或外观特征。

## 训练和判据

只训练四层零初始化残差 Adapter，原始 DriveEditor（包括旧空间/时间注意力）全部冻结。固定160步、AdamW 1e-4、原 diffusion loss、seed6201；320×576，10帧，与既有短窗任务对齐。选择最终160步，不能按真实 DEV 挑 checkpoint。保留40/80/120/160步恢复点。

固定12评测例×3臂=36窗，576×1024、seed42、25steps：关闭 Adapter / 同一 Adapter 输入全未知 / 正常 O/N/U/Q。第二臂是输入消融，并非另训等容量模型。输入、H、alpha 写回和 seed 相同。真实无去车 GT；收益必须看幻觉车是否减少、真实后车与邻车是否保住，训练 loss 或合成洞 MAE 下降不足以认定成功。单帧不能认证时序；后续 HTML 保留同步视频和原生输出。人工 verdict 留空。

A022包含此前r21入口修复已改善的情形，应看是否回退，不能把旧入口收益算到新分支。A042原目标身份仍有输入疑点，保留结果但不能单靠它建立主要成功结论。

CPU 实际240帧条件合同、隐藏 X/Y 替换不变性、零头梯度和 Adapter off 等价检查通过；本次四通道版本尚无真实网络 GPU 前后向检查。首次开卡必须先完成该检查，再执行训练。没有创建定时任务，也没有挂起会自动抢 GPU 的控制器。

## 条件覆盖：测量值，不是收益

| case | split | scene | 洞内O | 洞内N | 洞内U |
|---|---|---|---:|---:|---:|
{table}

N 很稀疏或为0的例仍保留，不能把无证据当确定背景，也不能用这些例单独否定“有效背景条件是否有用”。小样本/粗框先验只能验证第一阶段接口与初步任务价值。

本次8例真实DELETE洞内N仅0–0.157%，O为0–18.209%。因此首轮对保护车区域的提示更充分，对整片背景约束明显不足；A022不能作为充分背景条件的否定例。已实看全部8例固定f5的原图／遮洞输入／条件局部对照，确认目标擦除入口与条件是独立层；粗O的外形、隐藏实例身份以及时序效果没有据此认证。保留对应 `r45/real_QA_0.jpg` 与 `real_QA_1.jpg`。

## 入口

远端代码 `scripts/worldsim_v77/target_protected/iteration13/`；run `{O}`。CPU审核页本地 `outputs/v77-onuq-r45/index.html`。

开 GPU 后运行（本次未执行）：

```bash
/root/autodl-tmp/envs/driveeditor/bin/python scripts/worldsim_v77/target_protected/iteration13/experiment.py train
/root/autodl-tmp/envs/driveeditor/bin/python scripts/worldsim_v77/target_protected/iteration13/experiment.py evaluate
```

工程入口和条件准备完成不计作真实 DELETE 收益。failure_ledger_delta：更新同一 V77-F02 的条件几何误标证据，不新增 failure ID。
'''
    backup=O/'before_changes/docs';backup.mkdir(parents=True,exist_ok=True)
    paths=['docs/RESEARCH_STATUS.md','docs/EXPERIMENTS.md','docs/research_failures/entries/V77-F02.md']
    for rel in paths:
        dst=backup/Path(rel).name
        if not dst.exists():shutil.copy2(REPO/rel,dst)
    (REPO/'docs/v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md').write_text(report)
    (REPO/'docs/RESEARCH_STATUS.md').write_text('''# 当前研究状态

更新：2026-10-03。分支 `research/worldsim-v7.7-target-protected-editing`，主机 wm-3090-1001。

用户收敛为 O/N/U/Q 小条件分支：先验证“哪里有车／背景／未知”对真实 DELETE 的价值，主干冻结，身份系统、学习state encoder、2DGS/surfel后置。停止扩建旧数据工厂。

当前 task `WS-V77-TARGET-PROTECTED-20260929/r45` 已完成 CPU 准备：12旧训练例、4合成开发评测例、8固定真实DELETE；14个缺失LiDAR文件已补齐。O是GT框保守内核、N是排除实体后实测背景返回、U保留未知、Q为启发式可信度；不声称自动估计或精确分割。GT相机/车辆框为POC辅助，Y仅监督。初始整框O误标道路和不必要地面拟合限制N的对照均保留；统一用内核O与背景返回N，未逐例调规则。

157888参数四通道Adapter及240帧CPU合同通过。新训练0步，真实网络四通道GPU前后向未执行，模型收益未验证。固定160步，只更新Adapter；随后12评测例×3臂=36窗，比较关闭分支／全未知／正常条件，不选择性更换评测例。人工verdict空。

用户最新要求：先完成CPU准备，通知后再开GPU。当前无GPU训练/推理，也无等待自动启动GPU的控制器或定时任务。本地条件预览 `outputs/v77-onuq-r45/index.html`。细节、覆盖比例和组件图见 [r45报告](v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md)。稀疏N与粗几何是本次POC的明确边界；真实DELETE跨scene收益仍未达到。

旧Q060条件正例、r7等全部历史权重及反例保留。磁盘仍约44GiB可用，本轮没有清理旧文件。沿用 [V77-F02](research_failures/entries/V77-F02.md)。下一步由用户开GPU后先验证冻结/零初始化，再跑有界训练与固定真实评测。
''')
    p=REPO/'docs/EXPERIMENTS.md';lines=p.read_text().splitlines()
    if not any('20260929 / r45' in s for s in lines):
        i=next(i for i,s in enumerate(lines) if s.startswith('|---'))
        lines.insert(i+1,'| WS-V77-TARGET-PROTECTED-20260929 / r45 | O/N/U/Q小Adapter；CPU就绪，训练0步，等待GPU | [报告](v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md) |');p.write_text('\n'.join(lines)+'\n')
    p=REPO/'docs/research_failures/entries/V77-F02.md';body=p.read_text()
    if '## r45：粗条件入口与最小Adapter' not in body:
        p.write_text(body+'''\n\n## r45：粗条件入口与最小Adapter（CPU准备）

整块GT车辆cuboid投影即使固定内缩2px，仍在M042覆盖邻接道路；它是包络而非实体轮廓。初始条件图保留；本轮在所有case统一只用保守内核O、边缘U、Q=.5。N取同窗可见LiDAR正证据，排除所有实体包络及source H；去掉不必要的仅地面拟合限制，包含实测墙、护栏等背景，不作无依据稠密铺面。粗条件可能仍有几何误差，不能认证为逐像素真实标签，也不能将没有O的区域当N。

复用12训练/4合成DEV/8真实DEV，补14小LiDAR文件；157888参数Adapter的CPU合同通过。正式训练与新模型推理均0，尚无条件分支价值的科学结论。用户要求CPU完成后通知开GPU，不自动启动。下一步固定160步主干冻结训练与36窗三臂对照，不再先扩完整身份/surfel或数据工厂。见[r45组件图与报告](../../v77/TARGET_PROTECTED_ONUQ_ADAPTER_R45.md)。failure_ledger_delta: updated V77-F02（条件几何误标入口），无新ID。
''')
    print('RECORDED_CPU_READY')

if __name__=='__main__':main()
