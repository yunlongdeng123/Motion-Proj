# V43 Revision5本地发布包

该目录只准备文档和报告骨架；未执行SSH、Git提交、push或GPU工作。

## 本地准备

```powershell
python work/v43_gt_protocol_20261030/publication/prepare_publication.py
node work/v43_gt_protocol_20261030/publication/render_report_template.cjs
python work/v43_gt_protocol_20261030/publication/validate_publication.py
```

`payload/`包含Revision5独立页面、两张组件图、合同、设计源、当前状态快照及R4本地原始快照。归档文件为`publication.tar.gz`；其`PUBLICATION_MANIFEST.json`枚举全部可写文件。远端原有Revision4文档和代码保持原路径，发布脚本还会在campaign保存发布前的远端R4归档。

默认状态快照明确为“r2数据16帧验证通过、GPU smoke待跑”。发布前若真实状态已推进，使用`--status-file`传入主代理核实过的当前快照，避免旧快照覆盖新进展。

## M0骨架与最终报告

`m0_report_template/`只供检查和复制。最终报告拟放到`outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030/`，本工具没有写该路径。主代理根据真实数据、GPU smoke、完整恢复和编辑评价补齐HTML与`report_state.json`，将`artifact_kind`改为实际报告状态，并保留未完成项。不要将骨架中的“待填”替换成未经证实的通过。

默认发布包不包含M0骨架。要纳入已填写的报告，显式提供：

```powershell
python work/v43_gt_protocol_20261030/publication/prepare_publication.py --m0-report outputs/V43_Ownership_20261027/gpu_v43/sim_mechanism_20261030 --status-file PATH_TO_CURRENT_STATUS
```

报告将映射到`docs/autoresearch/v43_ownership_20261028/sim_mechanism_20261030/`，并在EXPERIMENTS追加独立M0入口。骨架目录和`artifact_kind=skeleton_not_final_report`会被拒绝。

使用`--m0-report`时，同时将本地`work/v43_sim_mechanism_20261030`顶层的Python、JSON、CJS和Markdown，以及`simulation/`和`reconstruction/`中的同类源码／配置／README归档到`scripts/worldsim_v82/v43_ownership_20261028/sim_mechanism_20261030/`。不打包PNG、模型、临时outputs、缓存或`__pycache__`；清单明确列出`m0_source_files`，远端仍按逐文件allowlist暂存。

## 远端使用入口

由主代理传输`publication.tar.gz`和`publish_remote.py`到远端专用`publication_r5`目录，再先运行dry-run检查具体文件范围：

```bash
PY=/root/autodl-tmp/envs/motionproj/bin/python
PUB=/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028/sim_mechanism_20261030/publication_r5
"$PY" "$PUB/publish_remote.py" --bundle "$PUB/publication.tar.gz" --dry-run
"$PY" "$PUB/publish_remote.py" --bundle "$PUB/publication.tar.gz" --stage
```

脚本替换RESEARCH_STATUS同一快照，EXPERIMENTS仅按task ID去重追加。`--stage`按manifest逐文件`git add --`，保存暂存diff统计并执行`git diff --cached --check`；不采用`git add .`，不提交或push。已有无关暂存会保留并使`--stage`停止；无关未暂存`run_worldsim_v72_e3_render_bridge.py`保留原字节。提交前仍由主代理检查完整暂存diff，完成提交和push核验。

发布脚本只写文档清单和实验索引，不启动训练、恢复历史队列或执行电源操作。
