# WorldSim V7.1 terminal launchers

这里保存 V7.1 第一轮已经完成并关闭的单次 orchestration wrapper：S1、S2、processed recovery、M0、M1 与 B4。

这些 shell 文件只负责激活环境、选择冻结配置并把输出重定向到 launch log；实际实现、配置和 canonical artifacts 仍位于：

- `scripts/run_worldsim_v71_*.py`
- `scripts/build_worldsim_v71_actor_corpus.py`
- `scripts/recover_worldsim_v71_actor_corpus_from_processed.py`
- `configs/worldsim_v71/`
- `/root/autodl-tmp/runs/worldsim_v71/`

归档原因是上述分支均已按 Source Selection stop rule 得到终态，不应再作为活跃研究入口误启动。需要复现历史 run 时仍可直接从本目录调用；后续 V7.1 诊断和新实验使用独立、显式命名的 runner。

本次归档不移动 Python 实现、不删除 canonical run/data，也不生成 hash、checksum 或 fingerprint。
