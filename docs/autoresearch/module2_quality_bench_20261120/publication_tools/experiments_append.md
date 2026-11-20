## 2026-11-20：道路资产细节、覆盖与静态交接复评

`MODULE2-QUALITY-BENCH-20261120 / road_quality_bench_20261120_r1`：[报告与组件图](autoresearch/module2_quality_bench_20261120/report.md)、[轻量结果](autoresearch/module2_quality_bench_20261120/results.json)、[评价协议](autoresearch/module2_quality_bench_20261120/protocol.json)。只复评冻结M01/M11等道路资产，不训练或更新参数；同分辨率稳定道路R/α为主、旧预乘空间为辅助，完整pre-alpha域保留。1920×1280真实新相机与贡献分解共12次原生前向，51.65秒，峰分配约4.93GB；CPU指标、0更新梯度探针及其他只读诊断分别记账。

原生细节未匹配：raw110/86孔洞σ1像素HF对应度.0268/.00065、能量比24.64/40.88。世界材质输入已存在主要不匹配；RGB与HF梯度弱对齐，教师box4控制另确认分辨率传递损失，均不构成表示上限或新loss有效性证明。完整孔洞的道路覆盖、mesh缺口及静态遮挡分别失败；全标线真值未建立。新增可复用域安全指标13项合成测试、2项HF数值/伴随测试，完整真实输入与大结果留远端，来源代码和小收据选定归档。未接受当前画质、未晋升默认场景，不复跑训练、不切换生成模型；failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06，human_verdict/problem_solved=null，GPU保持开机。
