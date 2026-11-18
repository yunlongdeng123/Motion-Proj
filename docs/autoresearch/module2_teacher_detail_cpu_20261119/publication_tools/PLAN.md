# CPU 发布执行计划

仅发布 CPU 准备来源与原生 adapter/通用运行时，不授权 GPU，不接管默认场景。所有实际 stage、validate、commit/push 和工作文档应用由 root 执行。

- 发布 parent：`58b60ea32e130daea14869e82c62feb7f91538ba`。
- 活动 HEAD 保留：`e10b5c20545a2737bb799bd7f7693c3e9e7e28cc`。
- 归档根：`docs/autoresearch/module2_teacher_detail_cpu_20261119`。
- 选定源码：Q/code 最新三个模块及三个测试，平铺到 scripts/worldsim_v82、tests/worldsim_v82；Q 其他执行源保留原字节到 run_code。小 JSON/模式文档归档，大数组、模型与媒体留远端。
- 当前快照只写本轮 CPU 状态；working candidate 保留并行二维任务原段。EXPERIMENTS/F06 保留工作树原字节后追加，发布版从指定 parent 追加本轮内容。failure 索引只在 scratch 再生。
- 隔离验证选择 root3＋permission1＋pixel1＋T2 1 共六个最终测试文件，预期 37 项。历史失败快照只归档，不收集其测试。Python3.10 CPU 测试；3.10/3.9 编译检查。CUDA 隐藏、线程 1。
- 私有 index 与 source_overrides 逐项绑定；默认 stage-only。diff 检查仅忽略 blank-at-eof，科学源末尾空行不改。远端 branch 若已推进则保存新 HEAD/重叠清单后停止，由 root 处理，绝不重置。

```sh
Q=/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1
PY=/root/autodl-tmp/envs/motionproj/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$PY "$Q/publication/tools/prepare_publication.py" --canonical-frozen --templates "$Q/publication/tools" --output "$Q/publication/prepared_r1"
# root 审 prepared_r1/{plan.json,source_list.json,working_docs.patch,publication_docs.patch}
$PY "$Q/publication/tools/publish_selected.py" "$Q/publication/prepared_r1/plan.json"
$PY "$Q/publication/tools/validate_publication.py" "$Q/publication/prepared_r1/plan.json"
# root 审 stage/staged.diff、source_validation.json 后显式决定 commit/push。
# 发布后再带备份应用；若任一原文改变，硬拒绝，不覆盖别人的新增内容。
$PY "$Q/publication/tools/apply_working_docs.py" "$Q/publication/prepared_r1/plan.json"
```

上述 prepare 仅在 root canonical 冻结确认后执行。source_validation 不能把 CPU 结论提升为真实 native 或最终画质通过。GPU 开启后的实际生命周期与配对执行仍为未来任务。
