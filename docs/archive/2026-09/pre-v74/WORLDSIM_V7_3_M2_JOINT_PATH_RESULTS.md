> 历史归档（2026-09-10）：以下为原版本事实与指令快照；当前执行以 docs/RESEARCH_STATUS.md 为准。

# V7.3 原生几何与空间查询联合通路

`WS-V73-M2-JOINT-GEOMETRY-PATH-01` r1=`done`，run=`20260907T151700Z__joint-dpt-query-s7302-r1`，code=`c0e23d28`，seed7302。12视图、1536查询、13824顶点/12288三角面，8步联合训练17.61s；峰值5.358GiB、RSS2.016GiB。原生DPT project梯度每步非零，参数最大变化7.758e-5；build center coverage从0.07640降至0.04688m。详表=`docs/autoresearch/worldsim_v73/m2/r1_summary.json`。

本结果只确认几何位置→局部视觉读取→预训练DPT的有效联合通路和当前配置资源；未训练patch法向/弯曲、无free/event、无held-out曲面比较，不能声明补全/主方法成功。`visual_observed_fraction=0.985`仅指投影在图内，尚非遮挡可见性。cKDTree避免全体两两建图，checkpoint保留12视图；不据此外推更多视图或上层LoRA显存。

下一步以同一显式曲面接真实束free/coverage与新时刻读出。当前不需要资源停机，也不把这一规模当作上层适配的资源上限。
