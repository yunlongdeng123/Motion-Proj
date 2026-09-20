# PIVOT failure ID

[全部 ID](../IDS.md) · [版本阶段记录](../VERSIONS.md)

每个 ID 优先链接独立卡，否则链接原定义；只有引用时明确标注。多条历史用 `python scripts/query_research_failures.py --id ID --detail` 读取。

| ID | 类型 | 首选证据 | 原标题 / 说明 |
|---|---|---|---|
| PIVOT-F01 | 历史标题定义 | [RF0915](../history/part-070.md)（行104） | PIVOT-F01：nuScenes cut-in 没有可验证的召回率分母 |
| PIVOT-F02 | 历史标题定义 | [RF0916](../history/part-070.md)（行129） | PIVOT-F02：贡献漂移——工程系统吞噬了重建与编辑研究 |
| PIVOT-F03 | 历史标题定义 | [RF0917](../history/part-070.md)（行146） | PIVOT-F03：未完成 exact reproduction 前禁止集成式“改进” |
| PIVOT-F04 | 历史标题定义 | [RF0918](../history/part-071.md)（行7） | PIVOT-F04：不能把“可见性建模”泛化成未观测背景已经解决 |
| PIVOT-F05 | 历史标题定义 | [RF0919](../history/part-071.md)（行30） | PIVOT-F05：资源不足时研究停机规则跨路线继续生效 |
| PIVOT-F06 | 历史标题定义 | [RF0920](../history/part-071.md)（行37） | PIVOT-F06：旧机器 smoke 证据不能替代新实例复验 |
| PIVOT-F07 | 历史标题定义 | [RF0921](../history/part-071.md)（行46） | PIVOT-F07：非 login shell 的 PATH 不能作为 CUDA provenance |
| PIVOT-F08 | 历史标题定义 | [RF0922](../history/part-071.md)（行53） | PIVOT-F08：在线浮动模型不能进入 exact reproduction |
| PIVOT-F09 | 历史标题定义 | [RF0923](../history/part-071.md)（行62） | PIVOT-F09：tar 页缓存与 nuScenes auxiliary 都属于资源/资产合同 |
| PIVOT-F10 | 历史标题定义 | [RF0924](../history/part-071.md)（行74） | PIVOT-F10：AD-GS 的 PNG 输出与 SAM2 的 JPEG-only 枚举不兼容 |
| PIVOT-F11 | 历史标题定义 | [RF0925](../history/part-071.md)（行86） | PIVOT-F11：COLMAP 默认全核并发会越过本机 cgroup 内存门禁 |
| PIVOT-F12 | 历史标题定义 | [RF0926](../history/part-071.md)（行103） | PIVOT-F12：跨场景连续执行时，official render 可先于 OOM 触发 cgroup 90% 合同 |
| PIVOT-F13 | 历史标题定义 | [RF0927](../history/part-071.md)（行120） | PIVOT-F13：processed scene 复用校验必须区分关键产物与合法空占位 |
| PIVOT-F14 | 历史标题定义 | [RF0928](../history/part-071.md)（行133） | PIVOT-F14：容器实例重建必须与 OOM/方法失败分开 |
| PIVOT-F14B | 历史标题定义 | [RF0929](../history/part-071.md)（行145） | PIVOT-F14B：pointops2 的 PEP 517 build isolation 没有继承已安装 torch |
| PIVOT-F15 | 历史标题定义 | [RF0930](../history/part-071.md)（行159） | PIVOT-F15：AD-GS 冻结 pseudo ID 与 checkpoint 都不能支持单对象编辑 |
| PIVOT-F16 | 历史标题定义 | [RF0931](../history/part-071.md)（行171） | PIVOT-F16：instance-aware 与 driving edit 已被 2025–2026 工作直接覆盖 |
| PIVOT-F17 | 历史标题定义 | [RF0932](../history/part-072.md)（行7） | PIVOT-F17：DGGT 扩展构建必须同时固定 compiler、headers 和 Python 依赖上界 |
| PIVOT-F18 | 历史标题定义 | [RF0933](../history/part-072.md)（行24） | PIVOT-F18：原生阶段完成不应被后续评估依赖失败覆盖 |
| PIVOT-F19 | 历史标题定义 | [RF0934](../history/part-072.md)（行35） | PIVOT-F19：nuScenes devkit 反向索引与磁盘 metadata 不是同一 schema |
| PIVOT-F20 | 历史标题定义 | [RF0935](../history/part-072.md)（行49） | PIVOT-F20：CUDA 扩展 import 成功不等于包含当前 GPU 架构 |
| PIVOT-F21 | 历史标题定义 | [RF0936](../history/part-072.md)（行61） | PIVOT-F21：训练完成 checkpoint 与累积式 post-render 必须分开裁决 |
| PIVOT-F22 | 历史标题定义 | [RF0937](../history/part-072.md)（行74） | PIVOT-F22：外层 timeout 不会自动回收独立 session 的 GPU 子进程 |
| PIVOT-F23 | 历史标题定义 | [RF0938](../history/part-072.md)（行85） | PIVOT-F23：SE(3) 一致性容差必须覆盖 float32 往返误差 |
| PIVOT-F24 | 历史标题定义 | [RF0939](../history/part-072.md)（行93） | PIVOT-F24：冻结 heldout 资源门失败不能靠事后更换 renderer 或提高阈值挽回 |
| PIVOT-F25 | 历史标题定义 | [RF0940](../history/part-072.md)（行107） | PIVOT-F25：部署 profile 必须区分传感器原始尺寸与 checkpoint 原生加载尺寸 |
| PIVOT-F26 | 历史标题定义 | [RF0941](../history/part-072.md)（行120） | PIVOT-F26：checkpoint state key 不能冒充加载后的模型运行时属性 |
| PIVOT-F27 | 历史标题定义 | [RF0942](../history/part-072.md)（行135） | PIVOT-F27：最小预注册剪枝臂失败后不能事后补更小 fraction 或放宽质量门 |
| PIVOT-F28 | 历史标题定义 | [RF0943](../history/part-072.md)（行150） | PIVOT-F28：顶层 `named_parameters()` 不保证覆盖普通映射中的子模型参数 |
| PIVOT-F29 | 历史标题定义 | [RF0944](../history/part-072.md)（行170） | PIVOT-F29：无卡实例必须以 cgroup 内存为资源合同，不能读取宿主机 `free` |
| PIVOT-F30 | 历史标题定义 | [RF0945](../history/part-073.md)（行7） | PIVOT-F30：原子发布目录不能把 `.partial` 绝对路径写进 manifest |
| PIVOT-F31 | 历史标题定义 | [RF0946](../history/part-073.md)（行15） | PIVOT-F31：SAM2 `reverse=True` 默认从最早 prompt 开始，可能合法地产生零帧 |
| PIVOT-F32 | 历史标题定义 | [RF0947](../history/part-073.md)（行22） | PIVOT-F32：mask QC 必须在同一像素坐标系比较 |
| PIVOT-F33 | 历史标题定义 | [RF0948](../history/part-073.md)（行29） | PIVOT-F33：大规模 Gaussian 重复索引累加不得使用逐元素 `np.add.at` |
