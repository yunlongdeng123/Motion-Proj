# 当前研究状态

2026-09-30，v77；`WS-V77-TARGET-PROTECTED-20260929/r3`。用户已开GPU继续造数，本轮冻结SAM2与新合成完成；原DriveEditor架构不变，训练0、模型补景前向0。[当前报告](v77/TARGET_PROTECTED_GPU_R3.md)。

4张真实多视角静帧轮廓通过，但居中窗口仅1/4合格；保留控制后，按全段几何/曝光有界选窗得到3/4，V002仍退出。3段30帧SAM2，独立连续mask通过3段。使用原官方缺少可选_C孔洞后处理的实际输出，未隐藏该运行边界。

新合成24例 / 5个receiver scene；类型{'background': 11, 'single_actor': 13}；独立{'pass': 24}，人工未填、训练准入0。五项训练输入标准生效；全类别洞/边缘扩张不得进入底部64px自车保守区，所有新增影响编码前被mask清除。旧37个1分重评中的27个技术候选继续单独保留，D006/D009不升级。

新多视角供体在两种放置控制均无可行提案；实际新合成来自此前已审干净来源的几何配对，排除8个旧重复提案，不能把结果算作新多视角素材收益。本地`outputs/v77-target-protected-r3/index.html`与`data_review.html`提供本轮报告和新逐帧页，旧r2页面/原AI分数不覆盖。GPU与CPU本轮作业结束，未启动训练、自动化或关机；不继承旧审计电源授权。后续来源必须先验证与receiver的空间/视角可配对性，再补RGB/SAM2；补密集类型及场景缺额，技术通过仍需用户全帧确认。subagent默认gpt-6-sol/xhigh，禁fast。

run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r3`。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
