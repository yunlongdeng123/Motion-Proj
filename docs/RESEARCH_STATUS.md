# 当前研究状态

2026-09-30，v77；`WS-V77-TARGET-PROTECTED-20260929/r2`。最新研究标准聚焦**训练真正看到的masked-X**：合理hole形状/尺度位置、时间连续、A→B正确遮挡和完整遮住synthetic影响；车身材质/受光/贴片感不单独否决。[当前报告](v77/TARGET_PROTECTED_CPU_R2.md)。

4份外部AI CSV49例810帧原样归档。用户最新明确授权重审原37个1分例，覆盖此前不重审/低分一律淘汰规则。一个Sol xhigh、非fast独立看首中末，630帧CPU全检查：{'train_usable_pending_human': 27, 'uncertain': 9, 'reject': 1}。技术可用仍待原定人工全检，训练准入0；D009用户明确空间失败保留0，D006也有已知ego覆盖。2分不自动准入。

官方get_blank及26字段CPU适配验证确认：完整synthetic-X不作为条件，先清零H再VAE/CLIP，真实Y仅作监督。630帧RGB泄漏0；latent nearest边界alias不等于RGB泄漏。未做VAE/UNet前向、微调或修改架构。

新供体入口防止旧污染mask复用；旧训练条件按五项逐例评估，不因供体RGB假一票否决。完整track真实多视角预检已选2实例4图（T021/S024），RGB齐；只待冻结SAM2小样，需要用户开GPU后继续。当前没有新合成视频、未达约50例合格目标。后续先验证新mask，再扩连续/多相机样本，不新增外观生成器。

本地报告`outputs/v77-target-protected-r2/index.html`，新判据人工页`data_review.html`；旧数据/外部AI评分全部保留。CPU任务收口，未设自动化/未关机；不得继承旧审计电源授权。run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929/r2`。subagent默认gpt-6-sol/xhigh，禁fast。failure_ledger_refs: [V77-F02]；failure_ledger_delta: updated V77-F02。
