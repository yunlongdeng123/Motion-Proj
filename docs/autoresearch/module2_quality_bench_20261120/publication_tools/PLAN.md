# 本轮 benchmark 的受限发布准备

当前仅准备，未私有暂存、提交、推送或更新工作文档。最终报告与结论由 root 冻结后再生成实际覆盖树。

## Git 边界

- 仓库：`/root/autodl-tmp/motion_proj_v82_gn11_v42`。
- 活动 HEAD：`e10b5c20545a2737bb799bd7f7693c3e9e7e28cc`，保持不动。
- 2026-11-20 只读 `ls-remote` 头：`48c6cca429ca5da5cb2f28ce2571190d38e01747`，分支 `research/worldsim-v8.2-drivingeditor-v44`。最终 parent 必须重新读取 origin，不以本地旧 ref 为准。
- 所有 Git 读操作设置 `GIT_OPTIONAL_LOCKS=0`。实际暂存仅用 `GIT_INDEX_FILE=<prepared>/stage/publication.index`；不使用共享 index，不 checkout/reset 本地分支，不改别人提交。
- 远端若推进，保留新父所有内容；重新准备/审阅本轮受限路径。提交前若又变化则停下报告新头与路径重叠，不能强推。

## 精确选取

可复用模块候选：`scripts/worldsim_v82/drivingeditor_dynamic_edit/road_detail_metrics.py`，来源 B/metrics 的冻结原字节，SHA256 `86780162b5bb3ab87e455054c800fccdeaba12337557960870b1b4943e92a03c`。

新测试：`tests/worldsim_v82/test_road_detail_metrics.py`。现有 dynamic_edit package initializer 会导入场景，故测试仅加 direct-module-directory 的 sys.path 前导，原13个测试正文及bare import原字节保留。原冻结测试另原字节归档；接入差异单独记录，不改 package initializer。两个目标现均不存在，最终仍重查，存在未知不同内容时硬停。

归档根：`docs/autoresearch/module2_quality_bench_20261120`。

- canonical：root 冻结后的 `report.md/results.json/architecture.svg/protocol.json`，结果必须是轻量汇总；单文件上限2MB，超限不截断或冒充完整结果，改用 root 明确的 compact 产物。
- `run_code/<B-relative-path>`：本轮B内的Python来源，原字节保存；包含本轮工程失败版本，但不运行其旧测试。不复制父G或其他二维任务历史。
- 配置/收据：采用 `selection.json` 明确白名单，保存metric freeze、13项指标测试/合成负控、1920读取资格/真实调用计数、CPU梯度结果、source-only结构候选边界、固定频率诊断与其他失败分层。
- 不选：33MB `lowres_r1/summary.json`、11MB `native_detail_r1/summary.json`、逐case完整tile JSON、NPZ/NPY/PT、模型、PNG/WebP/TAR和大读取缓存。这些保留远端，报告/README给精确复现路径。

不选 AGENTS.md、新旧二维报告、用户render bridge、既有persistent loader或其他dirty/untracked文件。若 root 给三文档模板，publication status 仅更新本轮3D快照，working candidate 保留完整二维段；EXPERIMENTS/F06从真实发布父追加本轮内容，工作候选保留原字节前缀，failure导航只在scratch再生。最终应用由root带备份执行，当前不写。

## 最终授权后的顺序

1. root 冻结 canonical 与选定代码/收据，再运行prepare生成 `tree/source_list/plan/publication_docs.patch`。准备器不创建private index；准备前后比较active HEAD/shared-index原字节。
2. root 审selected paths、来源和文档diff；publisher 只读ls-remote确认真实远端与plan一致，要求父commit已在本地对象库，不做fetch、不动任何local ref；private read-tree后只更新明确路径，运行`diff --cached --check`。仅忽略执行源末尾空行（`-blank-at-eof`），其余空白错误仍拦截。
3. 从private tree独立导出：CPU NumPy/SciPy 13项接入测试＋2项HF算子伴随测试；source原字节及测试prefix-only适配核验。Python3.10 CPU运行，3.10/3.9编译新source，不执行任何native/训练/关机入口。
4. root 最终确认后才 commit-tree(parent=实际origin)与普通FF push，并核远端head；active HEAD、local研究ref和共享index保持。
5. root 根据working-docs候选与before SHA带备份合并。并行二维段或任何文档已变化时重新合并，不能覆盖。

本步骤不会启动 GPU 或复跑真实 benchmark；GPU 保持开机。CPU/nativerender/gradient诊断与历史训练预算的科学结论以 root 最终报告为准。
