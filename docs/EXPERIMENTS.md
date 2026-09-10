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
