# 教师细节：CPU 来源与复现

[报告与组件图](report.md) · [结果](results.json) · [协议](protocol.json)。

CPU_PREPARED_STOP_BEFORE_GPU。当前 GPU/native/场景优化/新推理均为 0；默认场景不变。未来四组主对照 2800 物理更新，次要部分投影 TRELLIS.2 HF 初始化最多另 600，均未执行。

`run_code` 保留 CPU 准备及原生 adapter 的原字节；原生入口本轮未执行，默认拒绝未授权调用。仓库 scripts/worldsim_v82 的三个平铺模块使用相同源码，提供固定读取、共同世界材质运行时及执行硬门。不安装控制器，不接管默认 scene。

完整数组、权威场、算子和 CPU 数据夹具保留远端：`/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1`。其中 `permission/final_r1/teacher_evidence.npz` 与 recipe 为共同输入；平均算子为 `permission/avg_sparse_gh31_r1/operator`，逐像素算子为 `pixel_render/read_operator_box_r1/{086,110,134,179,198}`。平均算子 row_ids 是原完整道路拓扑 global world ID，须显式映射 active positions；K_H 已含 valid/mH，不得二次乘门。

最终六个测试文件由 publication plan 显式选择，只运行最终版本，不把失败工程快照的旧测试混入。验证在私有暂存树的独立导出中进行，CUDA 隐藏、1 线程；Python3.10 跑 CPU 测试，3.10/3.9 做源码编译检查。数据夹具可从冻结远端绝对路径读取，活动仓库代码不参与测试导入。

publication_tools 提供 prepare → stage-only → source validation → root review/commit 的受限流程；working docs 应在 root 发布后带备份应用，保留并行二维任务段与全部原实验历史。
