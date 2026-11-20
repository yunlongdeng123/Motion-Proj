# 道路细节benchmark来源

[报告与组件](report.md) · [结果](results.json) · [协议](protocol.json)。

本轮重新评价已冻结资产；不由指标更新推断训练收益、盲测结果、标线真值或表示上限。原生1920与480像素指标、稳定道路颜色与legacy预乘颜色分别保留。

run_code保存本轮实际Python源原字节；唯一可复用模块复制到scripts/worldsim_v82/drivingeditor_dynamic_edit/road_detail_metrics.py。CPU消费者请将该目录放入PYTHONPATH后直接import road_detail_metrics，避免现有package initializer加载场景依赖；未修改任何默认scene/initializer。测试只添加路径前导，原13个测试正文未改，另2个HF伴随测试独立验证。

只选明确小收据；33MB lowres/full tile JSON、11MB native摘要、所有NPZ/PT/图像/缓存保留远端：`/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1/gpu_20261119/quality_bench_20261120_r1`。复现入口为run_code/root_code，真实HF/edge指标和梯度诊断各有冻结输入、单位与分母。原训练属于先前研究，本轮CPU求导无参数步；任何原生读取调用由实收据单列。
