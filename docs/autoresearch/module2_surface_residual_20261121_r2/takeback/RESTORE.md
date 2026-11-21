# R2道路资产与阶段恢复状态

本包包含最终 appearance_fit/asset 的完整 road.npz、provenance.npz、scene/proof/static gates，以及 appearance、coverage 完整参数/Adam/RNG状态。coverage_fit/asset 是外观状态的精确基底；dense_init/asset 是覆盖状态的精确基底；另保留固定proposal、train_inputs、1191行许可与原R1复用caps映射、合法监督域和执行代码。

按 checkpoints.json 校核绑定后，从对应保存基底构造同一原生上下文，再加载完整状态；不要混用最终颜色资产作为外观checkpoint基底。未在此次取回中调用GPU或重新执行恢复。R1父资产与原大场景依赖按dependencies清单保留；本包不把107个复用行记为新增。

这不是完整独立scene备份。原静态、动态、环境大模型、原始图像/掩码与相机、教师监督输入等不复制，必要绝对路径和SHA见 dependencies.json。持久化manifest保留原路径；迁移时须使用明确的路径映射及proof/gate重绑定，不能单独手改scene.json。无默认画质晋升。
