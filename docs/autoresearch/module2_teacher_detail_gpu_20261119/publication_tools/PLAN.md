# 原生教师细节实验：来源发布准备

当前只完成来源盘点、活动 HEAD/index 与工作文档备份；没有暂存、提交、推送或覆盖工作文档。

发布目标为 `research/worldsim-v8.2-drivingeditor-v44` 的实际 origin 头。`prepare_gpu_publication.py` 每次最终准备重新读取远端；本地同名 research ref 不是发布父提交，保持不动。

## 最终冻结输入

等待 root 完成 `gpu_20261119/{report.md,results.json,architecture.svg}`。母协议仍取 Q/protocol.json；GPU授权另取 gpu_20261119/authorization.json，保留历史 CPU 合同。结果的 human_verdict 必须仍为 null。

root 提供模板目录：

- `RESEARCH_STATUS.md`：基于最新发布父树的当前3D快照。
- `RESEARCH_STATUS_working.md`：root 审核的活动工作树候选，保留完整并行二维段；工具不会猜测或覆盖工作文件。
- `experiments_append.md`、`failure_append.md`：本任务新增段，沿原历史追加。F06 原失败保留，补充本轮教师高频与理想读取正证据、仍存在的限制。

## 受限执行流程

使用 `/root/autodl-tmp/envs/motionproj/bin/python`，`CUDA_VISIBLE_DEVICES=''`。

1. root 冻结后调用 `prepare_gpu_publication.py --prepare --canonical-frozen --templates <templates> --output <new-prepared-dir>`。这里只在 campaign 生成选定树与计划，重新绑定实际来源字节，不写共享 index。
2. root 审阅来源、文档 patch 和数据边界。`publish_selected.py <plan.json>` 默认只使用 GIT_INDEX_FILE 私有索引暂存，`diff --cached --check`，不提交；父提交有变化立即停止。
3. `validate_gpu_publication.py <plan.json>` 从私有树独立导出，复用37项CPU测试加6项完整状态测试。唯一M12补充分支另含5项wrapper测试，最终期望48项；prepare显式传 `--extra-test source_filtered_training/code/test_source_filtered_field.py --extra-test-count 5`。检查实际源码原字节，两个Python环境编译；不使用活动工作树导入，不执行native/GPU。
4. root 最后审阅后才可用 `publish_selected.py <plan.json> --commit`：commit-tree 指定真实远端父、普通FF push，核对远端commit；不移动活动 HEAD、本地research ref或共享index。
5. 工作文档仅由 root 最后按 `working_docs`、当前字节及备份显式合并。并行改动发生后应重新准备，不能覆盖新版本。

归档仅代码、少量JSON、Markdown、报告组件SVG与索引；不包含NPZ/PT、纹理、像素数组、PNG或大型缓存。权威世界场、稀疏算子、输入和失败现场留在冻结远端。

本轮前置原生开销：失败恒色资格20 forward；完整资格82 forward +2 backward probes；理想读取30 forward；合计132 forward、2 backward、0优化更新。正式/生命周期开销必须另外从最终实际receipt取值，不能将前置调用归入正式步数。

本工具本身的盘点不代表科学验收、人工画质通过、默认场景晋升或GPU关机授权。

原五组3400物理/4000逻辑正式预算和结果保持不变。唯一前瞻性M12从原pixel共同A200分支，固定来源足迹过滤unmodulated H，再做B300/C300，额外600物理更新；不是原组延长、滤波扫参或TRELLIS扩展。sigma及regularizer语义由独立合同绑定，原五组源码不改。最终canonical应在M12与零更新诊断均收口后再冻结；新工程smoke及完整保存/恢复/下一阶段入口单列，不并入上述132次前置原生调用或17次原smoke更新。
