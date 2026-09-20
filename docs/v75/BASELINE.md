# V7.5 单卡正式 clean 基线

task/run：`WS-V75-BASELINE-01 / 20260920-single3090-r1`。工程开发结果；当前执行状态见 [RESEARCH_STATUS](../RESEARCH_STATUS.md)，研究协议见 [PROBLEM](PROBLEM.md)。

**单张 RTX 3090 已完成原配置 704×1280、30fps、237帧的正式生成及完整重复，没有 OOM。** 此结论限定于公开 single-view 2B 模型、预编码输入和本次约8秒序列。

```mermaid
flowchart LR
    A[官方 scene / 真实初帧 / 文本] --> B[官方条件渲染与输入编码]
    B --> C[条件数组 / 缓存 embedding]
    C --> D[官方 OmniDreams 2B 单卡生成]
    H[已生成历史与 cache] --> D
    D --> H
    D --> E[237帧 clean 视频]
    E --> F[完整重复 / 解码比较 / 输入输出并排图]
```

## 设置与资源

- 官方源固定 `bc711d6f95693d73693e6b5fac75749cc6fcf1d7`；本次入口实现基于项目提交 `ecbf5bdf`。第三方源码未修改。环境与模型来源见 [PREFLIGHT](PREFLIGHT.md)。
- 一次性文本和初帧编码已分别运行并释放；生成过程只加载生成所需模块，使用官方 `initialize_cache_from_embeddings` 接口。没有降低图像分辨率、缩短默认历史窗口、量化权重或改变调度器。资源结果以这一加载安排为条件，不能推出所有官方入口均可在24GB运行。
- 一个公开 scene `0d404ff7-2b66-498c-b047-1ed8cded60d4`，seed42；原始 actor/map/轨迹，动作锁定，未注入定位误差。初帧时间已对齐，只有一个真实RGB帧，无未来RGB真值。
- 独立进程执行首段5帧、完整30段、完整30段重复。合计61个生成段，属于1个场景/1个seed，不能当作61个独立样本，也不是61次底层DiT调用的计数。

| 执行 | 段 / 帧 | 生成循环耗时 | PyTorch峰值分配 | 整卡采样峰值 | OOM |
|---|---:|---:|---:|---:|---|
| 首段试跑 | 1 / 5 | 4.40s | 12.75GiB | 未采样 | 无 |
| 完整 clean | 30 / 237 | 100.83s | 12.81GiB | 17.41GiB | 无 |
| 完整 clean 重复 | 30 / 237 | 101.17s | 12.81GiB | 17.41GiB | 无 |

首次完整执行含模型加载与退出约126s；完整重复约126s。生成循环耗时包含同步、取回结果和逐帧编码，不是纯CUDA kernel时间。整卡显存每200ms采样，可能遗漏采样间瞬时峰值；PyTorch分配峰值与整卡占用口径不同。当前GPU足以继续这一配置的短序列实验，未检查更长序列或多视图。

## 输出与重复性

两个完整视频各237帧、1280×704、30fps、7.9s；全部帧成功解码且时间戳递增。固定抽查0、1.93、3.93、5.90、7.87s；道路场景随相机前进，条件中车辆与生成主体可对应。外观与细节仍有生成失真，尚未量化为状态误差或归因于重建误差。

完整重复逐帧比较解码后的RGB：237/237帧完全相等，MAE=0、最大通道差=0。范围是同机同环境下这一次重复及8-bit解码结果；未保留原始浮点输出，不能声称跨设备或浮点逐位确定。

单独5帧文件与完整文件前5帧的解码MAE为1.393/255。两者编码序列长度不同，存在视频编码上下文混杂，不能据此判定模型随机性；用于噪声判断的是同长度完整重复。保留该诊断原始记录。

人工verdict仍为null。当前结果证明工程可运行与一次完整重复性；没有证明普遍badcase、误差放大或闭环驾驶危害。未来真实RGB缺失，不能将该视频直接用于米制重建准确性结论。

## 复现与证据

```bash
cd /root/autodl-tmp/motion_proj
source scripts/worldsim_v75/environment.sh
python scripts/worldsim_v75/run_baseline.py --execute --blocks 30 --seed 42 \
  --output /root/autodl-tmp/runs/worldsim_v75/WS-V75-BASELINE-01/NEW_RUN/clean-seed42.mp4
```

必须使用新输出路径；脚本拒绝覆盖。原始记录：`/root/autodl-tmp/runs/worldsim_v75/WS-V75-BASELINE-01/20260920-single3090-r1`，包括三段运行日志、视频、两次整卡显存采样、退出码及解码检查。目录内的 `run_full_clean.sh`、`run_repeat_clean.sh` 保存实际启动命令；`inspect_baseline_video.py`、`compare_baseline_videos.py`、`make_baseline_preview.py` 保存生成报告的程序。

轻量证据见 [baseline](../autoresearch/worldsim_v75/baseline/)；其中解码与比较摘要省略逐帧数组，通过 `full_evidence_path` 指向保留的完整JSON。并排视频 `condition-and-clean.mp4` 和固定五时刻图 `condition-and-clean-contact.jpg` 位于原始记录目录。左侧为输入条件，右侧为模型生成；没有将条件图或生成结果标记为未来GT。

failure_ledger_refs：V75-F01、V74-H2-F20、V74-H2-F21、V74-H2-F22；failure_ledger_delta：none。当前没有新增科学失败或资源失败。
