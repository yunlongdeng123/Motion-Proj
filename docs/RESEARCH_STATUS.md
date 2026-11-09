# 当前研究状态

更新：2026-11-09（Asia/Singapore）。

当前任务`WAYMO-VAL029-TWO-EDIT-CASES-20261109`已完成。用户明确澄清：scene-029同一街段需要两个case，删除红车与删除后方白色SUV；scene-065是独立街段。此前错误地把原生来源删红车与当前场景删红车并列为两个主视图删除列；现已补齐实际白车查询，而非只改标题。[同帧图、组件图、源码与验收](autoresearch/waymo_val029_dual_delete_20261109/index.html)。

复用既有scene029完整10K、语义归属、白色SUV资产、道路与交接参数，优化更新0。红车为a69/native70（lvxXvuSB3kueTyP2ignh7w），后方白SUV为a24/native25（DwjZr_bYUSZOr-XDGregnA）。四帧130/155/160/190真实GPU pilot后导出109–197共89帧KEEP/DELETE_RED/DELETE_WHITE。全89的KEEP及删红与旧float结果最大差0，白车删除的白车组最大贡献0，红车身份行未过滤；灰车未被指定删除。目标组清空不是画质通过，删白车后背景没有独立真值，也没有为此另外拟合。

主视图五列为实拍、原生无编辑完整重建、共同KEEP、Case1删红车、Case2删白色SUV。原生来源删红保留在折叠的方法对照中。所有主视图渲染均为原PNG；原视频、阶段图、几何控件和既有结果保留。四锚帧补齐白车float32，贡献随case读取；删除白车时不拿原白车锚点深度评价新背景。诊断记录v2写明deleted_track并拒绝目标不匹配，兼容既有v1记录。实际viewer.js的DOM/Canvas替身测试、快速拖动、资源/PNG/float来源验收通过；没有浏览器视觉验收，未绕过此前file协议限制。

60,968,960B、275文件原始结果归档已取回逐SHA核验；新查询位于/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val029_dual_delete_20261109_r1，原模型与全部旧现场不改写。共享平台scene029仍为唯一入口，scene065及其他街段保留。failure_ledger_delta=none，本轮是查询补全与可视化修正；human_verdict=null。未使用subagent，无电源操作，用户不授权自动关机，GPU保持开机。

此前[scene065原生10K与归属/道路负例](autoresearch/waymo_val065_gpu_20261109/index.html)及V82-F06证据保持；不由本轮补图改变其质量结论。源码/证据按本次提交归档，推送与取回收据在用户工作区保存。不从历史队列自动追加训练。
