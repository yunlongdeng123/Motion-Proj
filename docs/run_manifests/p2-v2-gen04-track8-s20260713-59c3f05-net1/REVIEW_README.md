# P2-V2 人工复核

每个视频依次为 `[Base | generated point tracks | GT-ego debug correction | self-estimated correction | observed flow/self mask]`。
只评第二栏 generated point tracks：点是否大多贴合可见、可追踪的图像局部，跨帧是否连续；若点系统性漂移到无关区域、跨帧跳变或明显把前景/背景混淆，填写 `no`，无法判断填写 `uncertain`。后面三栏是已阻断 static branch 的诊断上下文，不作为本轮 verdict 对象。
复制 `reviews.template.jsonl` 为 `reviews.jsonl` 后运行同一命令并增加 `--aggregate-only`。
