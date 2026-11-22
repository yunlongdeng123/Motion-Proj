# 隔离索引发布入口

最终config与白名单已冻结：959行为保守清理候选，结论partial_cleanup_no_final_acceptance；原a20/default不变。root执行远端发布；无需训练摘要，不运行模型。GPU电源不由发布工具处理。

1. 以`manifest.template.json`生成最终config：填写实际selected_sidecar的path/sha256/bytes、固定actor SHA、actual_result_kind和三个文档片段路径。新白名单`finalized=true`，pending_required_sections清空，仅列真实来源。
2. 本地运行`python publication/build_upload.py --config publication/config.final.json --whitelist publication/whitelist.final.json --revision final01`。生成payload ZIP、manifest与上传SHA收据，封包上限4MB。文件源逐项列明，不扫描图片或模型。
3. 上传publication下的Python工具（须含scene_publication_scope.py）及上一步三个文件至远端本任务publication目录。用真实Python路径`/root/autodl-tmp/envs/motionproj/bin/python -B`执行后续命令，proxy本轮为`http://127.0.0.1:46849`，在发布时按当前会话复核。
4. 执行`capture_preflight.py --output <本任务publication/preflight_final01> --proxy <proxy>`，读取最新v44 parent和精确当前状态片段。最终manifest的status_update.old_source绑定该目录的status_replace_before.md。
5. 执行`run_prepare_stage.py --manifest <manifest> --archive <zip> --receipt <upload收据> --output <本任务publication/prepared_final01> --proxy <proxy>`。只构造私有index与精确暂存树，完成来源绑定、Python/JSON/SVG/diff检查；不提交。
6. 审阅stage/staged.diff后执行`publish_selected.py <prepared_final01/plan.json> --proxy <proxy> --commit`，再执行`verify_published.py <plan> --proxy <proxy>`。普通push到v44；parent变化即停止，不force/reset/merge。只取回publication_receipt/final_verification/source_validation及必要资源收据，已有源包不必重复下载。

新归档根为`docs/autoresearch/a20_scene_residual_cleanup_20261122`。同一报告仅允许scope列出的joint报告生成器、UI、result和index更新；共享报告source_manifest自动更新报告绑定并追加parent谱系，原科学来源记录不变。原历史groups、导航、actor验证及final_iteration逐字段保护，其余旧归档和AGENTS对象锁存。共享HEAD/index与全部dirty文件哈希必须前后一致。

恢复与最终参数均取实际收据。本轮14组封存，120步控制CANCELLED_NOT_STARTED。1468/1500/1754/2006不升级默认；其他道路任务继续运行，GPU保持开机。
