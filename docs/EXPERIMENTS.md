# V74 方法竞争：有卡恢复，首次实现进行中

2026-09-10，执行基线 e782b2b4。用户已解除 P0 阶段终点。RTX 3090 24GiB、14 CPU、90GiB 内存；当前没有资源阻断，V74-F02 GPU 状态已解决。P0 结果保持，原始数据不重跑。

| task | 当前状态 | 实际证据 |
|---|---|---|
| WS-V74-RESUME-01 | done | start/resource_resume.json，分支干净、GPU 空闲 |
| WS-V74-FIT-CALIBRATION-01 | pending | 只用各自 FIT 固定几何/数值预算，DEV 质量尚未读取 |
| WS-V74-METHOD-TOURNAMENT-01 | implementing | WEX/RIF/DCS 独立实现及强控制；43 ready/8 日志 + 23 缺 BUILD 合同对象 |

主 seed=7401，确认=7402；输出≤4096三角形；各自 FIT/DEV。尚无模型质量、候选裁决或新科学失败。原 nuScenes FINAL 数据缺口 V74-F01 保留。
先建立成熟 OSQP/HiGHS 对照与统一真实求交，保存机制算例、初始/事件/最终资产及全部对象分母。详细失败见 [V74 失败过程](WORLDSIM_V7_4_FAILURES.md)。failure_ledger_delta=resolve V74-F02 GPU resource; no science verdict。

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
