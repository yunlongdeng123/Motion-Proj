# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 `wm-vgpu-1008`。

当前唯一run `WS-V77-TARGET-PROTECTED-20260929/r51`：按用户要求再筛20–40个真实DELETE输入，先区分原生生成失败与最终写回损伤。40个新scene候选经独立5.6-sol/xhigh图像复核，33景/33个清晰单车目标准入，7例低质量、0例不能确定退队列。与r50全部38个审核过的scene分离，旧5景train隔离和val隔离不动；来源仍是既有RGB缓存，用于开发而非最终泛化评测。

固定r46官方原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、训练或时间模块修改。每车独立原视频。B实际可见证据另记{'weak': 8, 'sufficient': 8, 'absent': 17}，没有隐藏区GT。用户指定r50原生失败11例与仅写回损伤8例已分别归档，R009混合待核；旧分数/全部对照保留，写回损伤不拿去训练diffusion。

CPU准备和330帧原视频解码完成，当前无GPU任务、SAM/DriveEditor各0窗、训练0步。已停在等待用户开启GPU；开卡后先SAM身份检查，再一次固定DELETE，保留四列原图/mask/原生/最终，按原生失败优先匹配造数。失败生成只能定位布局，未来Y必须真实视频，不自动微调。

本地 `outputs/v77-real-delete-r51/index.html` 当前是输入页，生成列留空。数据盘约余39.1GiB，本批预计新增≤3GiB，无需扩盘/清理；没有定时任务或电源操作。[本轮组件图、清单和入口](v77/REAL_DELETE_NATIVE_R51.md)，[V77-F02](research_failures/entries/V77-F02.md)。r50已完成的40个原生/写回结果及原有Excel评分保持归档。
