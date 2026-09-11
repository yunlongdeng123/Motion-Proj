# 历史原始记录 001

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

# V74 最终失败索引（2026-09-11）

**NO_SURVIVOR，冻结本轮。** 人工verdict空；[最终报告](WORLDSIM_V7_4_RESULTS.md)、[机器判定及run清单](autoresearch/worldsim_v74/final/decision.json)、[逐项过程](WORLDSIM_V7_4_FAILURES.md)。以下是各ID当前状态，后文旧阶段文字是演进历史，不重新激活已结束实验。

| ID | 类别 | 最终状态 |
|---|---|---|
| V74-F01 | 数据/独立确认 | nuScenes新FINAL身份仍缺；无PASS，未进入扩展 |
| V74-F02 | 资源 | 有卡恢复，已解决；本轮无OOM/资源停止 |
| V74-F03 | 工程 | WEX自由约束关联修复，原r1留档 |
| V74-F04 | 新颖性 | WEX FAIL_NOVELTY，标准控制解释真实收益 |
| V74-F05 | 工程 | DCS锚点队列遍历修复，预算/网络保持 |
| V74-F06 | 工程环境 | NKSR独立cu118环境/编译/权重完成 |
| V74-F07 | 工程 | WEX真实距离责任排序修复，正式r2完成 |
| V74-F08 | 数值工程 | RIF同目标松弛消元完成；AV2迭代上限仍标注 |
| V74-F09 | 科学/增量证据 | DCS FAIL_SCIENCE；合成正例保留，eta容量贡献未检验 |
| V74-F10 | 科学 | RIF FAIL_SCIENCE，覆盖收益伴随超限early |
| V74-F11 | 外部导出工程 | NKSR8个缺层对象修复，无成功子集筛选；原空面保留 |

failure_ledger_delta: F04=WEX FAIL_NOVELTY；F09=DCS FAIL_SCIENCE（局部机制正结果保留）；F10=RIF FAIL_SCIENCE；F06/F11=外部环境/空层导出工程项解决；F01=独立 nuScenes FINAL 身份缺口保留，因无晋级者未进入 FINAL。

---

## 历史阶段记录与 V73 结论（不覆盖）

# 统一失败账本：V74 当前入口

日期：2026-09-10；有卡研究恢复，执行基线 e782b2b4。本文件保留统一失败 ID、索引和结论，V74 事件详情见 [V74 失败过程](WORLDSIM_V7_4_FAILURES.md)；历史指令不产生当前执行授权。按 ID 渐进式读取，每里程碑同步状态/实验台账。

| ID | 分类/状态 | 事实与后续 |
|---|---|---|
| V74-F01 | data/protocol；active | 现有 nuScenes 68 trainval 日志均有历史角色，尚无可直接证明全新未曝光的 10 个日志；现有域内 FIT/DEV 可用 |
| V74-F02 | resource；resolved | 用户有卡开机，RTX3090 24GiB、14 CPU、90GiB，已恢复方法研究；P0 低配事实保留 |
| V73-F02/F03 | support/optimization；historical active | 覆盖与提前面冲突、缺正确支持；A/B/C 改变重建变量，不恢复 loss 网格 |
| V73-F04/F05 | composition/data；historical active | 框归属代理与缺输入/缺返回分母仍保留 |
| V73-F09 | model；historical active | 联合视觉开发收益与跨域反转保留，不外推全部域内方法失败 |

当前实现里程碑：FIT 定标已完成，WEX 解析域子系统与公共求交首次实现，尚无新科学结论。OSQP/HiGHS 用作成熟强控制，不将 QP/MILP 自身当创新；来源见 [方法实现记录](WORLDSIM_V7_4_METHODS.md)。

## V74-F03 / V74-F04：WEX 初始机制与 FIT 证据

V74-F03（engineering, resolved）：初版冲突关联遗漏 free 约束连接，15/20 条件问题未探索联合责任；加回同一超图的一跳连接后 WEX20/20。r1和r2都保留，首次实现调试不记科学失败。
V74-F04（support/novelty risk, active）：成熟MILP在解析20/20且更快；FIT4对象与WEX同命中/同面数，密对象的正确候选缺失约70%。尚无DEV质量，不能外推全部WEX表示失败。详见 [V74详细过程](WORLDSIM_V7_4_FAILURES.md)。

## V74-F05 / V74-F06：C 初始调度与外部环境

V74-F05（engineering, resolved）：C提议调度将所有高需求点截成前32，固定20轮反复访问；改为完整排序轮转，四FIT对象所有控制完成r2，原r1与代码保留。未改变候选预算/网络/训练目标，未读取DEV。
V74-F06（environment/network, active）：NKSR需要匹配nvcc的PyTorch；使用独立cu118环境，官方857.6MB wheel下载15s读取超时，改为curl续传/180s超时。不是算力或磁盘不足，其他方法正常推进。详见 [V74详细过程](WORLDSIM_V7_4_FAILURES.md)。

## V74-F07：A固定责任的候选排序更正

engineering；corrected code / rerun pending。FIT第23束证明正确带内候选存储顺序不是距离顺序（24.020634 vs23.774240m）。按t稳定排序后更正固定责任与共享控制，实际硬首交点评价始终正确取min。旧A资产保留且标记superseded，不根据其DEV分数调参；详情 [V74失败过程](WORLDSIM_V7_4_FAILURES.md)。

当前里程碑：96cef75c配置冻结进入真实DEV评价；F06安装处理中，F07/F08正式修正版完成后收口，无新增科学失败或人工verdict。

V74-F04最新：排序更正后的真实DEV表明WEX与固定/贪心责任的命中、early、miss、recall相同，nuScenes命中还低于A3 MILP0.182pp。新颖性增量不足，等待三维机制统一收口；人工verdict未填。F07对应修正资产/评价已完成。历史R8超过4096面，明确列在原生预算参考。

## V74-F10：RIF提高覆盖但留出提前命中增加

主域RIF100个子问题均收敛，真实相对B2 hit+6.599pp伴随early+4.820pp，超过允许+0.5pp；相对B1 early+5.930pp/free+0.125m，相对B0 hit−0.586pp/early+5.156pp。无工程缺输出或超面数解释该现象。BUILD16/25对象物理成立，不认证未观测空间。AV2部分迭代上限单列，不能扩大科学否定范围。详见 [V74失败过程](WORLDSIM_V7_4_FAILURES.md) 和 evaluation/b_rif；机制/外部参考待最终收口。

## V74-F09：DCS真实DEV未形成相对强控制的共同收益

quality/novelty evidence不足，当前冻结模型/预算；nuScenes相对C1命中−0.405pp、early−1.439pp、召回−2.528pp、free+0.004113m，未达两条改善路径且召回保护失败。AV2自身训练相对C1命中−6.039pp、miss+8.380pp。无空面/工程缺失可解释该差额；不能以相对C0增密收益称对偶需求创新。详细表与逐对象证据见 [V74失败过程](WORLDSIM_V7_4_FAILURES.md) 和 evaluation/c_dcs。机制与日志区间待收口，人工verdict留空。

## V74-F08：RIF显式松弛求解成本（已迁移，完成组仍在执行）

environment/numerics；原FIT密对象100k级松弛使RIF396s，且外层300s预算不能限制一次内部求解。代数消元保持原凸目标，L-BFGS-B只优化场系数；同大FIT粗网格目标651.864→628.004，计算10.719s。迁移同时用于B0/B1/B2，真实DEV未读。详见 [V74失败过程](WORLDSIM_V7_4_FAILURES.md)。

## V74-F01：nuScenes 独立最终身份不足

观察：V73 log_payload_inventory_r1 报告本地 27 可用日志；旧角色覆盖 trainval 全部 68 日志。角色分配不等于实际训练曝光，但不足以证明从未曝光。当前 20 FIT/5 DEV 共 489 对象已完整准备，DEV 不是新的盲测。
检索：nuScenes 官方 test 不公开本任务直接需要的对象标注；AV2 官方 Sensor 提供公开带标注日志及按文件下载（来源见 P0 报告）。
迁移：AV2 旧曝光 20 日志固定 ID 排序，首 3 DEV、其余 17 FIT；另预留/下载 10 个未见于既有身份引用和原始目录的 FINAL 日志。新 FINAL 原始载荷就绪，未读取标注/LiDAR 数值、未做规范坐标导出或模型质量评价。
边界与复开：nuScenes 严格 FINAL 需真实未曝光身份/数据证据；不改名旧 FIT/DEV、不把缺数据当科学失败。当前双数据集开发和 AV2 最终确认路线可继续。
证据：configs/worldsim_v72/data_roles.json；docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json；configs/worldsim_v74/{data_roles,av2_final}.json；P0 数据摘要。task=WS-V74-P0-DATA-01。

## V74-F02：P0 无 GPU 已解决，低配 CPU 预处理保留

恢复更新：2026-09-10 用户有卡开机，RTX3090 24GiB，cpu.max=1400000 100000，memory.max=96636764160；GPU 无其他任务，资源阻断已解除。证据 start/resource_resume.json。
历史观察：nvidia-smi 权限被拒绝；cpu.max=50000 100000，memory.max=2147483648；没有 CUDA OOM。宿主 112 核/755 GiB 不属于实例预算。
已迁移：按 PyTorch 官方 mmap 读取既有逐对象数据，不加载大 RGB 拼包；1425 对象导出 15.04s / RSS 0.375 GiB；局部几何缓存 20.50s / RSS 0.097 GiB。两数据集字段往返核对通过，未产生模型分数。
存储事实：已释放 131.802 GiB，P0 结束前剩余 194.126 GiB；5 处视觉前缀、旧逐帧传感器/语义缓存已删除，保留 340 个代表文件及关键 checkpoint/表面/原始数据/指标。历史逐帧分析可能需重建并恢复依赖，不能宣称零成本完整重放。
复开：数据/文档 push 后确认任务全部退出，shutdown 并提示加卡。GPU 方法规模和外部基线需求在有卡实例实测，不按当前低配机器削弱计划。
证据：docs/autoresearch/worldsim_v74/p0/{data_summary,geometry_summary,data_roundtrip,storage_plan,storage_result,handoff_summary}.json。failure_ledger_delta=CPU 规避完成，GPU 资源需求保留；无科学裁决。

---

# V1–V73 历史事实（冻结；不作为当前执行授权）


## V7.3 最终科研收口：独立联合确认失败（2026-09-09 21:42 UTC）

V7.3/Q-v2及最终可训练视觉重接实验已结束；最终联合可泛化物理表面假设未获支持，按用户情况2收口。开发5日志hit+10.536pp等收益保留；固定20日志AV2 Actor确认r7−r6的hit−2.942pp、early+5.364pp、free+.325997m、距离+.013259m、recall−2.420pp，五项95%日志区间均不跨0且变差。不能外推所有视觉基座无效，也不把跨域失败冒称同分布新日志必然失败。

最终场景确认done：20日志40 heldout帧×6方法=240记录，每方法3908250原始束。cohort返回r7−r6 hit−2.211pp（[−3.253,−1.136]pp）、miss+1.347pp（[+.542,+2.432]pp）、returned MAE+.223034m（[+.136486,+.323606]m）；边界带hit−1.578pp、free+.034947m、returned MAE+.191461m的区间也均不跨0且变差。全束hit−.1845pp/miss−.1695pp均不跨0，其余全束三项区间跨0；背景分母较大不能遮盖Actor退化。边界与归属为box proxy、背景不完整、未知区域不当FREE，F04未解决。CPU wall720.392s、RSS1.207802GiB，0优化更新，无重复评价/汇总。

证据：docs/autoresearch/worldsim_v73/final_confirmation/（Actor code6499d34d，已push 9b6bc02e）；final_scene/{analysis,summary,manifest,compact}.json（scene code65854c27）。原始checkpoint/显式表面/逐束输出保留在runs/worldsim_v73相应run中，报告列出原路径。paper/main.pdf更新为17页，包含原组件图、真实Blender失败切片、最终外部与场景结果；LaTeX编译成功、无未定义引用/溢出警告，新增页已渲染查看。

30分钟自动跟进worldsim-v7-3已通过应用接口暂停，并核实本地status=PAUSED。所有科研计算结束，当前只执行最终文档提交/push与关机。GitHub push成功且最终进程检查确认无训练/评价/数据/渲染/汇总任务后执行shutdown；本节记录关机前交付状态，实际命令回执保存于本地outputs/V73_SHUTDOWN_RECEIPT.md。不得因历史计划或旧自动化prompt重新启动研究。

| 里程碑 | 执行状态 | 科学结论 |
|---|---|---|
| R10/R12/R14三角 | done | 覆盖/物理冲突，转入表示研究 |
| Q-v2闭合与开放chart、constructive/first-surface对照 | done | 表示与监督有局部收益，未同时解决物理一致性 |
| r6/r7最终LiDAR/可训练DPT联合比较 | done | 开发收益，独立联合确认失败 |
| 固定AV2五方法/同背景六方法确认 | done | 最终主假设rejected，不按确认结果调参 |
| 最终报告、失败记录、paper | done | 正负证据及原型/未验证提案明确分开 |

failure_ledger_delta=update V73-F02/V73-F04/V73-F05/V73-F09；科学风险未因项目结束解除，无新ID，下一V73-F10。无资源不足/OOM导致停止；停止原因是用户指定的最终实验边界已经完成。以下为历史快照，不恢复旧授权。

---

已证实的失败表现是跨域联合增量反转及early/free冲突；开发取证表明规则局部片也会错误延伸并遮挡后方正确支持，严重sliver并非这些early面的解释。域差异、视图/标定嵌入与米制对齐残差、局部视觉对应污染、支持域外扩是可能机制，未做因果隔离，不能写成已定位唯一根因。Joint同时改变native种子、可训练DPT特征及build-depth辅助项，故不是纯视觉特征消融。无OOM或资源阻断，不能用算力不足解释本次负结果。

---

## 固定20日志确认已启动，论文取证协同（2026-09-09 19:04 UTC）

WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1已于19:03:26 UTC启动，执行code6499d34d、shell PID169169。19:04:35快照Joint r7已完成68/936对象，allocated峰值5.437806GiB；r6/R8/native融合/PCA待顺序执行，0优化更新。外部模型质量评价现已开始，不能再写“20日志质量未读取”；配置已于开发阶段固定，禁止根据外部结果更改。完整五方法后收口Actor日志配对，再一次固定背景场景确认；当前无完整跨域结论。证据final_confirmation/started.json及run/manifest.json，日志controller_logs/final_confirmation_r1.log。

并行论文任务明确负责paper/、paper_forensics和自己的脚本，未修改本任务台账，也不会启动确认。已核实其取证summary/manifest：原75 DEV/11886 owned束，六方法AdaPoinTr、VGGT-native、R8、r3、r4、r6，不含最终r7；q=三边平方和/(4√3面积)，early面q>10计数全部0。r4/r6约94.7%/91.0%的日志等权early贡献来自“连通片至少有一个build端点在.2m内”的片。此支持定义不证明整个片正确，也不证明没有任何形变；不能将错误默认归因细长面或无支持浮片。按build支持剪连通片使r4 miss20.3879%→26.2698%、r6 20.5579%→27.4044%，并不能保住原覆盖。基于heldout删除early面是oracle诊断，未改部署输出或训练，不拿它作为模型成绩。证据paper_forensics/20260909T184200Z__saved-surface-oracles-r1/{summary,manifest}.json，已存在诊断不再重跑。

r7开发已显示hit/miss/distance增益而free风险仍在，保持完整结论，不宣布整个联合主线失败或物理问题已解决。论文的可视化和新算子原型归并行任务，其原型不是本次固定r7权重的方法，也不是新确认候选。Blender仍CPU渲染；最终shutdown需所有训练/确认/数据及论文渲染任务结束并完成push，当前不关机。

三本台账/最终结果报告同步，failure_ledger_delta=update V73-F02 with existing forensic evidence; no new failure ID。F03/F04/F05/F09边界仍按证据保留，下一V73-F10；每30分钟跟进至最终收尾。

---

## 最终联合r7收口：开发几何增益成立，物理风险仍在（2026-09-09 19:00 UTC）

最终开发对照r7−r6显示联合几何增益：hit+10.5363pp（95%日志配对区间[+6.4624,+14.6212]pp，5/5改善），miss−4.3793pp（[−9.3948,−.2924]pp，4/5），测量→表面distance−.079399m（[−.187384,−.009326]m，4/5）。early−1.5225pp（[−5.7790,+2.7510]pp）、free+.042022m（[−.000489,+.069738]m）、recall+8.4288pp（[−.7072,+23.9799]pp）三项区间跨0。free四日志变差，不能用不显著掩盖其风险，也不能把此结果归类为Joint≈LiDAR或全部失败。当前仅支持联合通路对开发几何与正确首返回有增量；完整物理主张和跨域泛化尚待固定确认。

WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7，训练code185bc728，30轮11130实际更新/0跳步/0恢复，489 initial/final完整。75 DEV/5日志，23无owned/8空对象保留；hit/early/miss/free/distance/recall=.496040/.214590/.161786/.279829/.093915/.808715。wall20249.418502s（5.625h）、GPU10.223834GiB、RSS34.127140GiB，744冻结前缀视图；可训练DPT32654562/Query1706413，DPT首project最大变化.005171027。原生辅助173972测量/轮，measurement-weighted Huber .845831→.486418m；几何通路有效不等于所有物理指标改善。

r7汇总已由并行论文任务执行一次（/root/autodl-tmp/paper_r7_summary.log），本任务复用归档，不重复汇总/训练。模型与确认方案已固定：按登记的FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1顺序运行r7/r6/R8/native融合/PCA，随后同固定背景场景评价；不根据新日志调参或再启动候选。当前20新日志的模型质量仍未读取，确认待启动。跨域和背景问题作为结果边界，不能预先宣布论文主假设全部成功。

另有用户正在进行的“更新WorldSim论文与失败切片可视化”任务，负责paper/和paper_forensics，已发送共享文件协调信息；保留其未提交文件，不代为stage/覆盖。它的Blender任务仍运行，最终shutdown必须等待其渲染/论文任务完成以及所有训练/评价/数据任务退出。当前任务继续负责最终确认及三本台账，最终push后再关机，不恢复无限扩展。

证据ray_support/open_joint_r7_{summary,final_manifest,analysis,training}.json及配对/训练/组件图。failure_ledger_delta=update V73-F02/V73-F09 evidence; no new failure ID，F03/F04仍未解除，下一V73-F10。

---

## 最终确认流程登记，r7正常训练（2026-09-09 13:50 UTC）

13:48只读快照：最终联合r7仍shared_train、第3/30轮，原生DPT与Query均有梯度，GPU占用12452MiB/95%利用率，allocated峰值10.223834GiB；没有失败或资源短缺，保持原配置。此为单步运行状态，不解读为质量改善，不重新执行梯度检查。

固定最终确认入口已准备，尚未执行或读取外部质量。WS-V73-FINAL-CONFIRMATION-01/20260909T135000Z__fixed-r7-r6-external20-r1，scripts/run_worldsim_v73_final_confirmation.sh：仅r7完整30轮及DEV收口后，顺序评价Joint r7、匹配LiDAR r6、旧窄片LiDAR R8、M1头native融合、LiDAR PCA五个固定方法。原20 AV2日志936对象/878 ready/58缺输入，28路时刻视图，所有缺输入与缺返回保留；同一已准备build-only IRLS米制尺度与标定/轨迹，不改camera映射、不打开新visual-only路径、不做新日志优化或挑epoch。主对照r7−r6，其他为强弱与表示参照；这是真正跨数据集确认，不能写成nuScenes同分布新日志。没有未来训练或网格搜索。

固定表面齐全后，WS-V73-FINAL-SCENE-01/20260909T135000Z__fixed-r7-r6-external20-r1，scripts/run_worldsim_v73_final_scene_confirmation.sh：一次CPU BVH组合四个保存模型表面（r7/r6/R8/native融合）及既有background_only/场景LiDAR PCA控制。同20日志、同build雕刻background.npz与逐束只读轨迹，字面全局首交点，不按heldout删除背景/Actor面。场景内置PCA控制与Actor表格的PCA分别按既有实现报告，不混成同一数值。r7−r6主配对保留，完整场景分组/所有原始返回保留；跨域和背景边界仍需按证据报告，不把编辑演示当真值。

分析复用已保存结果，Actor按独立日志汇总六指标及配对区间，场景按原始束汇总日志配对。不提前启动确认、不改运行训练；这次仅准备收尾入口和通用绘图role参数，没有新推理/测试。最终报告纳入正负结果、F02/F03/F04/F09及失败解释，三本台账和GitHub push完成、停自动调度且确认无任务后shutdown。failure_ledger_delta=none at preparation，旧风险active；30分钟跟进持续至收尾。

---

## 最终联合r7通路检查通过并启动（2026-09-09 13:15 UTC）

一次真实首个ready FIT检查done，code185bc728：24实际DPT视图，115原生框内种子候选、512completion seeds、无LiDAR fallback。仅几何项（coverage/free/box/首面）对四层DPT特征的梯度范数为.00122070/.00250244/.00194550/.00164032，对DPT首project参数梯度188.088562；这些数值未混入辅助depth梯度。辅助项读取3条build测量，.261523m；联合梯度范数789.206299，沿用clip1，一次Adam后DPT参数最大变化1.00285e−5。检查6.219805s、GPU11.874109GiB、RSS2.912842GiB，无DEV/新日志质量读取，scratch权重不复用；有梯度不代表泛化有效。

WS-V73-Q-V2-01/20260909T131000Z__open-charts-joint-first-surface-s7304-r7于13:14:02 UTC fresh启动，训练code185bc728/PID156599。13:15:35快照仍initial_evaluation，744冻结前缀视图、371 FIT/438有输入/51缺输入，共489对象，0正式更新；不能写成已完成initial或训练收益。完整初始后按30轮/11130预期更新运行，原生DPT约3265万参数与开放Query共同训练；不解冻aggregator、不用缓存DPT输出、不改变运行配置。新20日志质量仍未读。

主比较r7−r6；同表面/物理损失与更新预算，Joint新增native种子、可训练DPT特征与build-depth辅助监督，结论针对整条通路。完整final后仅执行一次scripts/summarize_worldsim_v73_open_joint_r7.sh，附旧闭合joint r1与R8；禁止中途挑epoch/根据新日志再调参。原始支持/物理风险仍active，不额外延长V7.3。

证据ray_support/open_joint_check_r1.json、r7_started{,_manifest}.json，简单组件图V73_OPEN_JOINT_COMPONENTS.{png,pdf}已生成并查看。failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F09]，failure_ledger_delta=none at start，下一V73-F10。三本台账和报告同步push，30分钟跟进；最终对照/确认/报告完成且无任务后shutdown，现在不关机。

---

