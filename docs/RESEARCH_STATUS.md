# 当前研究状态

更新：2026-11-22（Asia/Singapore）。

`A20-SD15-SAMEVIEW-20261122`：[车辆教师与高斯资产报告](autoresearch/a20_sd15_sameview_20261122/report.md)、[HTML对照](autoresearch/a20_sd15_sameview_20261122/index.html)、[实际阶段计数](autoresearch/a20_sd15_sameview_20261122/result.json)、[恢复入口](autoresearch/a20_sd15_sameview_20261122/restore_index.json)。当前部分完成，场景交接未过。SD1.5唯一真实同视角候选止于二维结构门，V/R硬保护误差0，未新增Omni；Amodal真实partial结构及测距未过；TRELLIS同HQ参考的对象旧数值门4.68/9.08cm通过，统一scale1.10触界。24k+235独立学生完成20smoke+2replay+1丢弃探针和正式40geometry，99%覆盖/HF未过，formal material0并封存。独立410080原生支撑保留控制完成5smoke+2replay及80真实可见DC校正，几何/Σ/opacity/未知DC逐位冻结，独立磁盘重载8视图float/8bit差0；不是完整压缩蒸馏。两组正式120、含短程全部150更新。旧行纯退休的K保护RGB/Alpha差0，新增对象仍遮挡保留车辆，raw86 native11变化239/245像素、raw110全K488；真实接地支持缺口/冲突及原生残留未解，未晋升场景。scene/road更新0；最终actor及checkpoint、core94与scene71文件、本地报告22状态QA已核验。human_verdict/problem_solved=null，failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06；GPU按用户要求保持开机。

免责声明：参考图及生成隐藏车身属于车型先验；1030点为对象框筛选的测距归属代理，BUILD86/110参与配准或观察拟合，134/179/198未参与本轮选择。单候选及不同输入协议不能决定生成模型或3DGS能力上限；当前未通过完整车辆真实性、接地、残影与保留车辆交接验收。
