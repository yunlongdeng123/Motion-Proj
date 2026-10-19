# GN11 四组消融＋原生五身份基线：AI review材料包

共 **4组新实验＋1个原生基线**，均为五身份a6/a37/a5/a26/a18，raw89–137完整49帧，逻辑第一阶段30K，第二阶段0。基线严格对应用户指定页面中的 **五身份GN11第一阶段30K**；不包含其后实例训练或20车B实验。

## 从这里开始

1. 阅读 `REVIEW_TASK.md`，再看 `experiments.json` 的开关和实际完成状态。
2. 打开 `index.html` 或 `compare.html`：五个配置的原生四通道完整49帧，可同步播放/逐帧、四组总览和与原生A比较。
3. `independent_pair.html` 是full与real_only已经完成的独立七通道。其余两组独立删车/alpha仍待GPU，不存在的结果没有补造。
4. 先读 `code/diffs/baseline_to_full.diff`，再读 `full_to_*`。完整原生核心代码只存一份在 `code/base/`，四组实际变更存 `code/overlays/`。`code/source_map.json` 提供每组实际文件hash。
5. `evidence/runs/` 按五组分开：真实运行配置/状态、24个新实验保存节点CPU审计、原生30K评估事件、实际前100条采样日志、已完成的独立评估凭据。

| 配置 | 归属修复 | 真实道路RGB权重 | 供体RGB权重 | 额外独立49状态 |
|---|---|---|---|---|
|baseline|关|0|0|existing_native49|
|full|开|1|0.1|completed_zero_update|
|real_only|开|1|0|completed_zero_update|
|prior_only|开|0|0.1|pending_GPU|
|ownership_only|开|0|0|pending_GPU|

深度权重为真实0.1、供体0.01，关闭时为0；没有重新分配或归一化。所有新组的static/dynamic/environment实际更新计数各30000。原生基线的历史status没有本轮新增的显式计数器，以其原生完成事件、模型节点元数据和恢复证据解释，不伪造计数器。

## 目录

- `code/`：原生核心、四组差异、实际入口/运行绑定、离线核查脚本和许可证；不含编译二进制。
- `evidence/`：按run分组的配置、保存/恢复凭据、冻结查询及49个真实道路训练目标、来源诊断和已做验证。
- `frames/`：raw89/129/137的评估GT PNG和五组完整画面样本，以及查询/供体全幅图；全部49帧GT保留在完整视频中。
- `media/`：37路完整49帧视频引用，按文件SHA256去重后29个实际MP4，另加1个基线固定相机时间诊断视频；均保留480×320完整画面，通道名称和来源独立保留。
- `MANIFEST.json`：逐文件SHA256/字节数；`PROVENANCE.json`：Git与范围；`VALIDATION.json`：打包检查。

## 必须区分的边界

- full＝归属修复＋真实曝光＋供体；real_only/prior_only也保留归属修复。ownership_only才是道路目标全关闭组。原生baseline没有本轮三类改进。
- 道路来源类别和目标颜色被冻结：7,787个同世界位置真实确认样本、5,119个异位置供体样本、2,963个未知样本。UNKNOWN从不监督道路；关闭一种来源不会把其区域改成另一种来源。样本计数不是均匀世界面积。
- a6的1,149个raw89查询射线当前无合格真实确认；245供体、904未知。a37的14,720个样本中，raw131–137确认4,765，全49确认7,787；供体4,874、未知2,059。没有确认不等于物理上永不曝光。
- 模型坐标锚帧raw99、道路/SGCM锚帧raw89，二者分开。供体框是干净沥青来源，不是query；白线、黄线、自行车标记边缘已过滤。
- 所有49帧都是训练重建视图，额外3D数据集框用于归属/遮挡约束；不是heldout，也没有隐藏道路GT。旧初始GPU张量快照不可得，只能核对输入/seed/采样协议，不能宣称逐位旧初始化相等。
- 原生基线曾因CUDA计时事件读取中断，保留20K并经明确授权重放丢失5,846步，逻辑终点30K；仅改同步/保存工程行为，没有本轮归属或道路目标。相关历史凭据附在baseline目录。
- 冻结config/bundle的 `execution` 字段是训练前的历史计划标记；当前完成状态以每组真实 `status.json` 和 `experiments.json` 为准。全部原始证据保持原字节，不为排版改写旧状态。
- 消融的effective_config/v2_policy保留冻结完整版目标描述；实际道路权重在各自overlay的v2_runtime_context.py中验证variant后覆盖。请同时核对实际CUDA gate的actual_road_weights和源码常量，不要只把旧配置里的完整版权重当作消融实际执行权重。
- full/real独立恢复模型、Adam、RNG、sampler、SGCM后零更新导出；prior在用户切无卡时中断，0PNG/无完成凭据；owner独立导出尚未启动。原生静态没有改名冒充独立删车。
- 实际全49静态观看仍见残留：full有矩形道路条纹/接缝；real有黄白亮块/蓝边；prior有黑块；owner有车状轮廓。未作干净静态质量通过判断，完整多通道质量复核未完成，human_verdict=null。

## 代码复原（仅复制，不训练）

```sh
python code/materialize.py --experiment full --output /path/to/new_empty_review_dir
```

此工具只重建/核对核心源码。实际入口保留远端路径；本包用于代码和结果review，**不包含全量数据、checkpoint/Adam权重、CUDA扩展和全部外部依赖，不是可直接续训的复现实验环境**。不运行任何入口也可完成静态review。

## Git和完整性

- Repository: https://github.com/yunlongdeng123/Motion-Proj
- Branch: `research/worldsim-v8.2-drivingeditor-debug`
- 代码快照与本指南同提交发布；最终ZIP的PROVENANCE.json记录确切代码提交。
- Remote source snapshot collected UTC: `2026-10-19T14:24:18.909584+00:00`
- 打包仅CPU只读取源，无新GPU/训练更新；内容hash见MANIFEST。媒体复用既有完整文件，没有再裁剪或生成补帧。
