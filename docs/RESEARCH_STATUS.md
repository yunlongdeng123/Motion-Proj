# 当前研究状态

更新：2026-09-29，v77。唯一任务`WS-V77-DELETE-AUDIT-20260928/r1`的45个nuScenes val scene、70个单目标DELETE均已完成并同步本地。25个final scene仍隔离，九个旧例仍为DEV；没有新训练、Ω/GLB推理或逐场景调参。人工0/1/2 verdict全部为空，等待用户视频审核。

同一冻结p2提示、SAM2、DriveEditor与写回规则完成210窗：198窗实际生成、12窗全空mask保留RGB。280段视频共7280帧已实际解码；本地70张卡片、280视频与700个资源引用均核对存在，MP4字节数匹配远端，JS语法通过。浏览器自动访问file URL被安全策略拦截，没有绕过，也没有声称浏览器交互验证通过。

每例只在冻结prompt审核一帧。粗分类：背景模糊/伪影22，本帧未见明显问题19，无法确定11，车体身份待核实9，真实后车结构异常2，疑似车辆再生4，mask缺陷3。它们是抽帧观察计数，不是视频成功率或纯模型失败率。45例本帧未见明显mask错、22例输入边界/身份不清、3例有mask缺陷；另有21个clip其他帧含GT投影存在但core为空，两种口径不能混用。助手难度high/medium/low为37/28/5；冻结代理31/33/6保留。

本地审核：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-delete-audit/index.html`，提供原视频黄框／模型mask／原生deletion／最终DELETE四列及评分导出。远端完整run：`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-AUDIT-20260928/r1`。原始RGB、mask、原生和最终PNG、初版及修正预览均保留。factual reconstruction保持未运行/不可判。

本轮推理和编码控制器已退出，GPU空闲。用户已授权实验完成后关机：当前完成交付、保存及推送后，最后核对无其他作业/控制器，再执行AutoDL关机；尚未执行的动作不记已完成。执行凭据与SSH断连核验将保存本地`outputs/v77-delete-audit/shutdown_receipt.json`。半小时监控`v77-delete`将在关机完成后删除。后续等待用户审核，不自动启动训练。

见[最终结果与组件图](v77/DELETE_AUDIT_RESULTS.md)、[执行记录](v77/DELETE_AUDIT_GPU_RUN.md)、[冻结协议](v77/DELETE_AUDIT_PROTOCOL.md)。本轮failure_ledger_delta为updated V77-F02（冻结审计观察及边界）。