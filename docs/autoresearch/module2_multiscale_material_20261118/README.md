# 多尺度道路材质：源码与复现

[结果与组件](report.md) · [结构化结果](results.json) · [实际执行账](execution_accounting.json)。

工程链路通过，质量联合门失败；默认场景不晋升。共享场1400正式物理更新，匹配独立RGB800另列。A为真实可见监督，C/S为冻结伪教师。

run_code保留实际科学执行源原字节；仓库scripts中的三个模块是可复用提炼版，二者角色不混同。完整资产、教师、模型、checkpoint、query缓存与浮点图保留远端：
`/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1`。本目录不上传模型、数组或媒体。绝对路径及来源记录见evidence；原199BUILD只继承。

复现主方法：在AutoDL drivingeditor环境运行远端code/run_material_pair.py；输入与授权由execution_contract、formal_launch_contract及frozen_contract限定。L800/ABC800的权威field需经field_native_io重query；公共road.npz是派生GS缓存，不是另一次独立RGB训练。matched I0与TRELLIS.2入口见run_code/qa与run_code/trellis2，均保留对应冻结合同，禁止把历史I0替代匹配控制。

CPU验证采用CUDA_VISIBLE_DEVICES为空、1线程和禁用pytest插件；publication/source_validation记录验证的发布树。
