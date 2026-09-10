# V74 方法竞争：有卡恢复，首次实现进行中

2026-09-10，执行基线 e782b2b4。用户已解除 P0 阶段终点。RTX 3090 24GiB、14 CPU、90GiB 内存；当前没有资源阻断，V74-F02 GPU 状态已解决。P0 结果保持，原始数据不重跑。

| task | 当前状态 | 实际证据 |
|---|---|---|
| WS-V74-RESUME-01 | done | start/resource_resume.json，分支干净、GPU 空闲 |
| WS-V74-FIT-CALIBRATION-01 | done | nuScenes 129812 / AV2 672669 跨 BUILD 平面比较，epsilon=.174520/.200000m；DEV 质量未读 |
| WS-V74-METHOD-TOURNAMENT-01 | implementing | WEX/RIF/DCS 独立实现及强控制；43 ready/8 日志 + 23 缺 BUILD 合同对象 |

主 seed=7401，确认=7402；输出≤4096三角形；各自 FIT/DEV。尚无 DEV 模型质量或候选裁决；V74-F03 初始冲突图遗漏已修复，V74-F04 登记 WEX 的标准求解器替代/支撑缺失风险。FIT 定标是各自数据内完成，局部平面残差含曲率/轨迹/归属代理误差，不是噪声真值；容差以上仍记冲突。原 nuScenes FINAL 数据缺口 V74-F01 保留。
OSQP 1.0.4 已安装，SciPy 1.15.3/HiGHS 可用。WEX 域子系统实现及其固定/贪心/MILP/软域控制已经写入；80 个解析约束算例 r1/r2 和两数据集各2对象 FIT pilot 已完成，不能代替完整三维机制与真实验证。公共硬求交/评价接入已有 V73 定义；保存机制算例、初始/事件/最终资产及全部对象分母。详细失败见 [V74 失败过程](WORLDSIM_V7_4_FAILURES.md)。failure_ledger_delta=resolve V74-F02 GPU resource; no science verdict。

WEX 实际证据：解析 multiple_front_constraints 20例中，r2 WEX20/20、贪心13/20、MILP20/20；MILP累计 .143s，WEX .230s。其余有正确候选的解析组贪心/MILP同样成功，不能声明机制不可替代。
FIT4对象中 WEX/MILP BUILD正确命中分别均为40/41、908/3195、74/74、1145/4208；两个较密对象缺正确候选2242/3009束，域选择无法凭空补齐。WEX计算 .575/45.30/.076/126.40s，MILP .060/.884/.089/2.471s；首个资产与V73首交点读出数值差0。无DEV/FINAL质量读取。
证据 `autoresearch/worldsim_v74/a_wex/{domain_r1_summary,domain_r2_summary,fit_pilot_results,fit_pilot_resources}.json`。failure_ledger_delta=V74-F03 resolved initial implementation; V74-F04 active hypothesis risk，尚无主方法晋级/淘汰裁决。

## B/C 首次实现与 FIT 训练里程碑

执行基线 b08b85d6，尚未读取真实 DEV/FINAL QUERY。RIF 的连续 P1 场/完整区间编译/星形一致细分与 B0 sampled、B1 exact、B2 普通几何误差细分已实现。80个固定场数值机制：false_free_claims=0，mesh–field根差≤4.77e−7m，延拓误差≤1.34e−15；这是数值/表示证据，尚不替代自适应与控制的端到端几何实验。FIT4 pilot 正在完成，首个41返回对象 RIF全中，B1/B2分别漏5/3束；较密对象仍有约束松弛，不作物理保证。

DCS 读取点/局部法向/BUILD射线及 LP alpha/beta/eta，PointNet局部编码输出中心位移、法向偏转、两轴有限尺度；真实八边形定价后做HiGHS整数选择，无透明响应。每数据集每FIT日志固定1个≥8点中位对象：nuScenes20日志/640教师例，AV2 17日志/544例；各自 full/no_demand 模型，均200轮，同初始化与更新预算，分别2000/1800次更新，每个模型14087参数（参数量以training.json为准）。教师只用本数据集 FIT BUILD + FIT额外射线，无DEV/FINAL。第一FIT对象八边形与导出三角的支持/early集合相同。

C 初始调度只轮转前32锚点，修复为在完整排序队列内轮转，固定20×8提议预算不变；r1/r2资产与原代码保存，V74-F05工程项已解决，网络没有重训/挑checkpoint。r2完整无需求/普通增密控制一同完成，BUILD安全仍伴随密对象支撑不足，待相同真实DEV评价，不先宣告胜者。

下一步：43对象/8日志三候选全部必要控制资产；80组三维真表面机制重建；共同强基线。NKSR官方源已下载，CUDA11.8独立环境安装中；PyTorch大文件15s超时已按官方超时说明迁移到续传与180s，V74-F06=环境/网络处理，不是GPU不足。failure_ledger_delta=V74-F05 resolved; V74-F06 active; V74-F04 remains unresolved。

## 固定资产进度与A责任排序更正

A/C各215资产（43对象×5方法）r1已完成，真实DEV QUERY尚未读。B四对象FIT仍在完成密对象求解，三维80例×14方法机制run已启动。公共r6/r7/R8原表面与4096面PCA surfel参考资产已准备172项；历史跨域训练边界单列。
V74-F07：FIT对象scene-0471__203cea9260874ff78e5e200a866ef44f第23束，正确带内首存储候选24.020634m，而最近候选23.774240m，观测23.871235m。初始A1把面遍历顺序当最近责任，违背对照定义。现按每束候选真实t稳定排序；A真实r1与三维run中的旧A保留但不用于正式裁决，A组将执行一次更正生成r2。物理评价器始终取min(t)，此错误只影响责任初始化/控制，不是查询器改分数。
无质量驱动调参，当前三候选参数/提议预算保持。新B/C/训练里程碑bdc10313已push。failure_ledger_delta=V74-F07 engineering correction pending corrected assets; V74-F06 install ongoing。

## RIF 数值求解迁移（DEV之前）

原显式松弛QP的FIT4×4结果全部保存。AV2密对象RIF耗时396.16s，超过300s外层预算：只有外层检查耗时，没有限制单次求解；100k级松弛变量使成熟求解器开销主导。按同一凸目标解析消去e=-Hc、s=max(mu-Fc,0)，用SciPy L-BFGS-B求剩余343–727个场系数，所有B控制同样迁移，加入求解回调时间预算。无约束删减/损失改权/新表示。
同两个FIT粗网格：小对象原/新目标−53.090999094/−53.090999374，大对象651.864226/628.003868；新解.529/10.719s。大对象原QP没有充分求到同目标的最优，不能把其失败当B科学证伪；新解完整H/F残差继续保存。V74-F08=数值成本问题，已迁移，完整FIT r2和43真实对象四控制准备推进。
旧80例混合机制run已因A排序/B数值版本被中断，保留已完成状态；r2只重建受影响A/B，已完成C几何结果原样复用，余下C继续。没有重复训练C。当前仍无真实DEV/FINAL结果或候选裁决。
NKSR主wheel续传完成；国内CUDA依赖改直连镜像，实际下载恢复约17MB/s。独立环境继续安装。failure_ledger_delta=V74-F08 numerical migration; V74-F07 corrected assets ongoing; V74-F06 progressing。

## 固定配置进入真实 DEV 评价

执行基线96cef75c。三候选、所有控制的参数与FIT checkpoint已冻结，首次真实DEV读取开始；此前没有读真实DEV质量。只评价已完成的固定资产，A取距离排序修正r2，B取同目标松弛消元版本，C取完整锚点调度版本。43 ready对象与23缺BUILD合同对象共同保留；5个nuScenes/3个AV2日志分别按对象内、日志内、日志间汇总，未知分母保持未知。
外部NKSR固定官方ks预训练权重，BUILD PCA朝传感器法向、原生0.1m体素、detail_level=0、mise_iter=1，不按DEV选择体素或裁剪面数。其原生面数/训练来源与4096面主比赛分开报告。安装尚在完成，不能把安装完成算作模型成功。
`autoresearch/worldsim_v74/evaluation/freeze_before_dev.json`保存配置和开始时间。failure_ledger_delta=none；F06外部环境仍处理中，F07/F08修正完整组继续，无候选裁决。

## C真实留出首轮结果：未达到共同晋级门槛

WS-V74-PROBE-EVALUATION-01 / 20260910__C-probe66-r1-s7401完成43 ready+23缺BUILD对象×5控制；ready没有工程缺输出/空面，11对象无自有QUERY返回仍在分母表中。nuScenes按5日志等权：DCS/C1普通增密/C3同网络无需求命中25.936%/26.342%/26.238%，early2.398%/3.837%/3.191%，miss66.777%/65.830%/66.741%；DCS相对C1召回下降2.528pp。DCS的early下降未达3pp，命中增长路径也未达3pp，自由侵入.006740m反而高于C1的.002628m，不能只报early改善。
AV2自身FIT、3日志DEV：DCS/C1/C3命中26.511%/32.550%/27.722%，miss63.402%/55.022%/62.320%，自由侵入.069451/.075572/.044875m。未要求跨域泛化。C相对无新增片C0改善，无法据此绕过强普通增密与去需求控制。V74-F09登记本轮质量/机制增量不足；最终候选分类待80几何机制与配对日志归因收口，不再按DEV改模型或预算。
B同目标消元FIT4×4已完成，无工程错误，完整原/新结果都保留。A/B正式资产继续。failure_ledger_delta=add V74-F09; V74-F08 corrected FIT completed; no overall verdict。

---

# P0 已完成实验

日期：2026-09-10；执行基线 463ef199；来源 V73 最终 01af4739。P0 只有数据/存储工作，optimizer_updates=0，model_quality_evaluated=false。

| task | 状态 | 实际结果与证据 |
|---|---|---|
| WS-V74-P0-DOCS-01 | done | 76 份归档；d6bea861 已 push；documentation_archive.json |
| WS-V74-P0-STORAGE-01 | done | 实际释放 131.802 GiB；463ef199 已 push；storage_plan/result.json |
| WS-V74-P0-DATA-01 | done | data/worldsim_v74/p0_r1；1425 对象，534.23 MiB（含几何/索引）；data_summary.json、geometry_summary.json |
| WS-V74-P0-DATA-01 / probe | done | 43 对象/8 日志，23 缺输入合同对象，11 probe 无自有 QUERY；probe_cohort.json 与 probe_availability.json |
| WS-V74-P0-DATA-01 / FINAL raw | done | AV2 新 10 日志/100 文件/65.79 MiB，缺文件 0；av2_final_raw_summary.json |
| WS-V74-METHOD-TOURNAMENT-01 | pending | 每数据集自身 FIT；主 seed=7401、确认 seed=7402；方法实现/训练、真实质量与裁决尚未开始 |

| 数据集/角色 | 日志 | 对象 | 有 BUILD | 缺 BUILD | 无自有 QUERY 返回 |
|---|---:|---:|---:|---:|---:|
| nuscenes FIT | 20 | 414 | 371 | 43 | 83 |
| nuscenes DEV | 5 | 75 | 67 | 8 | 23 |
| av2 FIT | 17 | 807 | 764 | 43 | 96 |
| av2 DEV | 3 | 129 | 114 | 15 | 23 |

CPU 实际工作：分离导出 15.04s / RSS 0.375 GiB；最终局部几何导出 20.50s / RSS 0.097 GiB；首次几何缓存修正为右手坐标基后重建一次。法向有效 1071227 点、退化 164 点；均为 BUILD 估计。
一次往返核对覆盖两数据集各一非空对象和一空对象，留出字段与 V73 原数组相同、query-ray 接口无距离真值、BUILD/QUERY 无重叠；data_roundtrip.json。未运行模型 smoke 或回归套件。
新 AV2 FINAL 仅原始窗口就绪，规范坐标导出与测试待方法冻结阶段；nuScenes 独立新 FINAL 身份限制保留，不把域内 DEV 计为未曝光 test。
failure_ledger_refs=[V73-F02,V73-F03,V73-F04,V73-F05,V73-F09,V74-F01,V74-F02]；failure_ledger_delta=update V74-F02 CPU workaround, no new science failure。
证据统一位于 `docs/autoresearch/worldsim_v74/p0/`。[V73 完整台账](archive/2026-09/pre-v74/V73_EXPERIMENTS.md)。
