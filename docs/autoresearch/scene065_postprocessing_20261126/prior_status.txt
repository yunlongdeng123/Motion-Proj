# 当前研究状态

更新：2026-11-25（Asia/Singapore）。

独立当前范围 `SCENE065-TEACHER-GAUSSIAN-REBUILD-20261125`：[组件、as-run 源码与最终证据](autoresearch/scene065_teacher_gaussian_rebuild_20261125/README.md)。原生未编辑30K保留场参数冻结，R7→R8→R9完成可变点数五参数Gaussian重建；道路累计2800、实拍背景出生后800个唯一更新，实际物理更新2804、额外replay4，最终新场69695点。20视图独立资产重载及16参考帧全部数组复现完成；车底残影与彩色边界仍未通过。A20继续采用原30K还是旧ab4fa资产待用户选择，本轮不自动替换车辆或追加训练。259个全20视图不可见的区域外静态核恢复未采用，0 optimizer，不计视觉改善。本段取代20261116同任务历史执行计划；旧现场与历史人工verdict保留，当前human_verdict/problem_solved=null。failure_ledger_refs=[V82-F06]，failure_ledger_delta=updated:V82-F06。

二维教师独立范围 `ROAD-2D-TEACHER-PROMPT-20261120`：[五seed提示词对照与组件](autoresearch/road_2d_teacher_prompt_20261120/publication.html)完成。用户已采纳整帧1152×768＋注意力LoRA T0为后续默认；本轮5seed×2prompt×6任务正式60次，另4次真实GPU smoke，4进程并行，训练与三维修改0。合成两底图等权ΔE76 2.239597→2.242254、共同频谱偏差1.201894→1.141373，但改善配对仅7/20与8/20，seed波动未降低；新prompt不晋升，不再追加调词。12次旧1152三图重放和实际W/K门通过，60原始PNG及全部对照资产已取回核验；真实隐藏道路无GT。保持历史活动提示词，原CPU/GPU报告及其他三维任务保留。method human_verdict=用户采纳当前审阅样例；prompt human_verdict/problem_solved=null。实例按用户明确要求保持开机。

[上一共享分支状态快照](autoresearch/scene065_teacher_gaussian_rebuild_20261125/prior_upstream_status_20261125.md)保留完整历史；历史阶段计划不替代本段当前范围。
