# 场景1身份与表示修复诊断

该包保留有限探针，不自动恢复历史队列，也不是分类器训练入口。`authorization.json` 是本次执行时的历史快照，不构成未来运行授权。实际入口读取工作目录中的授权文件；交付后关闭该文件的GPU执行开关。

运行环境：`/root/autodl-tmp/envs/drivingeditor/bin/python`，原生CUDA渲染依赖当前DrivingEditor环境。CPU拆分检查：`python -m unittest test_two_sided_split`。

脚本和数据的实际位置：

- 工作目录：`/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/editable_ownership_20261030`
- 原生渲染代码：同campaign父目录的`code/`。
- 输入合同：同campaign父目录`unknown_white_20261030/render_inputs/input_contract.json`，包含原static与Actor10K模型及数据源绑定。
- `scope.py`继承原R2允许范围；没有新增框外候选授权。
- 原始浮点、优化状态、运行日志及失败现场：同campaign父目录`editable_ownership_20261030_raw.tar.gz`，本地已核对取回。

`run_scene1_probe.py --help`列出显式输入合同、原生代码、范围和授权路径。`launch_probe.py`与`launch_refit.py`是当前现场便捷入口，路径固定；复现到其他位置应直接调用底层脚本并传入对应输入路径。输出必须是新目录。

三锚帧冻结探针新增训练0。raw113短拟合累计200步，只更新14个子高斯的DC与opacity；恢复r2新增0步。该优化拟合原模型输出，不能作为真实语义监督。`gt`在拆分代码中指平面greater-than侧，不是ground truth。新`r2_removed`图片是原环境来源对照，不替代原R2局部白底结果。

结果与边界见同版本的HTML报告。下一阶段需先检查/修复深度与支撑，再验证拆分及编辑收益；本包未实现最终几何修复或可学习归属模块。
