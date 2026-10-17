# R15冻结检查点诊断工具

这组工具只读取远端 `paused_checkpoint_004000_20261017/checkpoint.pt`，不调用训练恢复入口，不覆盖模型或优化器；生产SGD=0。使用原DrivingEditor环境和共享 `campaign/gpu_slot.lock` 串行运行。`audit_performance.py`对内存副本做F+B，Adam额外计时lr=0；`audit_actors.py`对49帧分层、46训练帧贡献及6锚帧做反事实；`audit_shadow_precision.py`核查分块、批处理、病态矩阵和FP64参考。

```bash
PY=/root/autodl-tmp/envs/drivingeditor/bin/python
D=/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3/backend_restore_r15
flock /root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock "$PY" audit_performance.py --output "$D/paused_diagnostic_20261017/performance_recheck"
flock /root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock "$PY" audit_actors.py --output "$D/paused_diagnostic_20261017/actors_recheck"
flock /root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock "$PY" audit_shadow_precision.py --output "$D/paused_diagnostic_20261017/shadow_precision_recheck"
```

路径在源码中明确冻结到本次campaign；复用于别的场景前须显式调整并核查输入契约。完整profiler事件的后处理可能耗时数分钟，不把该耗时混入同步吞吐。发布前对诊断脚本做了局部对象释放及key_averages后处理缓存整理，积分/渲染/损失公式未变。

`preserve_and_stop.py`是本次已完成暂停的操作记录：绑定已核实的PID/start_ticks，只用于这次进程身份，不可当作通用停止命令重复运行。controller.py的PAUSE_REQUESTED保护已实际运行验证，禁止自动恢复。原控制器保留启动step2断言，不能直接当作4k恢复控制器使用。

build_report.py / build_views.py用于本地工作目录的证据排版；路径需要按本地交付结构放置。报告中的稳定残差FP32、减少太阳样本、增大chunk、关阴影均为诊断对照，含已记录失败，不是已采用的训练修复。
