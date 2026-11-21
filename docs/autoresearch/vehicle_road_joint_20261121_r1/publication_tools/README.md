# 车辆与道路联合实验轻量归档

本工具默认只暂存。root 确认报告完成并给出最终 sourcefiles 后，再冻结清单与执行发布；实际结果以发布和取回收据为准。

目标分支为 `research/worldsim-v8.2-drivingeditor-v44`。报告与记录进入 `docs/autoresearch/vehicle_road_joint_20261121_r1/`，实际运行源码进入其 `run_code/`。只接受显式列举的轻量文本源；不含 PNG/JPG、NPZ/PT、模型或外部论文全文。

本轮正清单需包含实际 `frozen_formal_r2` 运行源码、lowerprior、handoff、QC/eval、最终报告源和 `link_existing_report.py`。旧 `work/road_surface_residual_20261121_r2/viewer.js` 的导航小改作为新目录下的 `linked_sources/road_surface_residual_20261121_r2/viewer.js` 保存；保留原源码路径绑定，不重新归档旧主页面的大型 HTML payload。工具拒绝 `reference/`、`hugsim_text.txt`、`old_report_before_vehicle_entry.html` 与旧 `target_visibility.*`。文件目录不会自动递归加入提交。

## 发布流程

1. 将确认的本地报告源上传到本轮独立 publication inputs 目录。保留原文件；清单 source 使用远端绝对路径。复制 `manifest.template.json` 为最终清单，填写 `sources`、最终状态快照片段与实验索引条目，最后将两个完成字段改为 true。
2. 检查当前 LocalTUN 会话并验证实际代理端口。当前只读检查中可用的是 `http://127.0.0.1:40051`；会话端口在正式执行时重新查询。
3. 在远端按下列顺序执行，使用独立且不存在的新输出目录。前三步不会创建提交。检查 `staged.diff`、`staged_stat.txt` 与 `source_validation.json` 后，只有收到 root 完成信号才执行带 `--commit` 的命令。

```bash
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python prepare_archive.py manifest.final.json --output /ABS/PATH/archive_r1 --proxy http://127.0.0.1:PORT
python publish_selected.py /ABS/PATH/archive_r1/plan.json --proxy http://127.0.0.1:PORT
python validate_archive.py /ABS/PATH/archive_r1/plan.json
python publish_selected.py /ABS/PATH/archive_r1/plan.json --proxy http://127.0.0.1:PORT --commit
python verify_published.py /ABS/PATH/archive_r1/plan.json --proxy http://127.0.0.1:PORT
```

4. SCP 取回 `publication_takeback.zip`、`takeback_receipt.json` 和最终收据；本地运行 `python verify_takeback.py publication_takeback.zip takeback_receipt.json --output takeback_verified.json`，核验 zip 与 `committed/` 内全部提交源后，再通知 root 取回成功。

清单条目格式：

```json
{"source":"/ABS/RUN/code/actual.py","destination":"docs/autoresearch/vehicle_road_joint_20261121_r1/run_code/actual.py","role":"run_source"}
```

## 共享状态保护

`prepare_archive.py` 从 origin 的已发布 parent 读取当前状态文档与实验索引，不从脏工作区读取它们。当前三维快照按唯一锚点替换，实验索引仅追加本轮短条；已有二维教师段落原样保留。每阶段核对共享 HEAD、index 与已有 tracked 脏文件。AGENTS 与二维教师归档对象受额外保护。

发布使用 `GIT_INDEX_FILE`、`read-tree`、`update-index`、`write-tree`、`commit-tree`。提交不切换工作分支，不改共享 index，也不复制其他脏改动。push 指向显式分支 ref，禁止 force push。parent 变化即停止并重新审阅，不自动 reset 或 merge。源码验收只表示冻结源与精确暂存树匹配，不替代科学、画质或人工验收。
