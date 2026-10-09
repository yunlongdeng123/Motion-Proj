# 当前研究状态

更新：2026-10-10 01:57（新加坡）。分支 `research/worldsim-v8.1-seen-to-scene`；主机 `wm-3090-1009`，checkout `/root/autodl-tmp/motion_proj_v81`。唯一task/run `WS-V81-SEEN-TO-SCENE-P1-20261009/r1`。用户授权自主完整复现、subagent审核、无human in loop；先排论文/工程差距再短周期验证。[协议](v81/YOUTUBE_VOS_P1_R1.md) · [审计/组件图](v81/P1_PROGRESS_AUDIT_R1.md)。

P0工程完成，非论文能力。旧AllFrames及公开reference_m4分别保存1000，传播方向/仅未来父链已由固定官方模块CPU反证，旧阶段hold、不放5000。reference_m4新增7窗完整25帧QA仍有结构错位/重复/滞后；visible与硬写回契约正确，模型画质未合格。旧控制器已停，全部断点/产物保留。

新paper_bidirectional_m4按论文Eq5/6目标→最近过去/未来direct flow、参考链组合、源mask与原FB检查，保留原_post_fuse参数结构；25帧最稀K42。按论文文字Adam/wd0，旧协议AdamW，断点不混用。首步通过，第2步OOM；原栈/step1保留后FCNet完整时轴激活重算、ternary按mask质量等价分块重算，从step1恢复成功。第2步allocated18.402GiB，三组件梯度有限/非零、冻结0；两步及1窗推理工程通过，独立审核仍见初始色块，不作方法否定。

接线独立复核无确定新bug、CPU全套75 passed后，仅恢复同协议至100做固定valid学习诊断，100后退出并等助手审核，不自动1000/100K。活动状态paper_bidirectional_m4/controller_state.json，scope在probe_100_scope.json；worker52384、trainer52387需实时核对。两步关键checkpoint硬链接preflight_checkpoints保留。不重复启动GPU、不停其他任务。

全视频入口/控制器/指标guard已接入：full-video全T参考条件、25窗口/stride16重叠均值、全局latent一次更新，VAE分块；正式清单必须source_dir与源/GT/pred/comp帧数一致。历史默认前25仅诊断，路径/缓存分开。上游RAFT/FCNet全T显存及全视频GPU尚待短检；作者预处理/FVD平均口径有未知，protocol_verified=false，正式四指标未算。

数据6包完整，不重下；train1951有效视频/19313窗口。DAVIS90+AppendixYT60固定，valid排除正式60ID；BUILD按公开GT flow/首帧完整CLIP，QUERY仅visible RGB。单3090/bf16/CPU卸载与论文资源差异明示，不加P2/P3、不载作者编辑权重。

审核页继续同步outputs/v81-paper-p1，保留历史输入/视频和CPU反证，补旧reference1000、新双向2步与独立QA。媒体/权重/外部源码/数据不入Git，源码ZIP上限100,000,000 bytes。当前不关机。failure_ledger_refs=[V77-F02]，failure_ledger_delta=none；这是工程/协议复现，非科学否定。
