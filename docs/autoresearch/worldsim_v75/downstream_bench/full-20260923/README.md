# 自建24-case：首轮推理与逐方法报告

Task `WS-V75-DOWNSTREAM-FULL-01`，run `20260923-r1`，seed42，单RTX3090。当前执行快照只见[RESEARCH_STATUS](../../../../RESEARCH_STATUS.md)。这是开发集问题发现，不是最终测试集排名。`failure_ledger_refs:[]`，`failure_ledger_delta:none`。

## Architecture components

```text
固定24-case + nuScenes事实RGB/标定/状态/地图
                 │
          单变量factual/CF适配
                 │
   ┌─────────────┼────────────────────┐
   │GT条件生成   │高斯重建+编辑        │下游consumer
   │Omni/ReSim   │StreetGS / HUGSIM    │GaussianDWM
   └─────────────┴──────────┬─────────┘
                           ↓
       同窗口 原始视频 / factual / counterfactual
                           ↓
         独立读出 → A/P/E/O/T/OP → HTML逐case报告
                      无总分；人工null
```

## 已执行与交付

- OmniDreams公开single-view 2B：24/24对，48段×77帧=3696帧，480次生成调用，生成阶段1583秒。每支前70帧(0–2.3s)计入公共窗口，末7帧状态冻结填充不评分。全部24对原始生成数组前13帧完全一致。权重复用，每支独立cache并重置seed42 RNG。
- 输入是GT地图/轨迹，不是高斯重建输出；6 native ego + 18 adapted actor。GT actor尺寸取全track中位数、未来GT状态用于控制，输入信息预算显式披露。事实/CF同初图、同文本、同非目标状态；地图双线以对应单线primitive适配，z=0。
- nuScenes pinhole到FTheta使用完整可见域5阶minimax近似；原先复用AV2的0.05px容差失败，保留0生成失败；新适配网格32769点残差<0.5px、三独立cuboid测试<3px。
- 全部24case有 `original-nuscenes.mp4` + `original-nuscenes.json`：1600×900、10Hz、24帧、不插帧，来源为nuScenes DriveStudio预处理JPEG，不声称原始相机比特流无损。与生成共用来源窗口/相机，0.5秒发生干预；不是反事实GT。
- HTML总览新增“反事实任务”列，每组视频前新增中文任务卡：减速/加速倍率、横移幅度、移除对象、插入供体与局部偏移，以及事实/CF要求、生效时间、保持条件和预期可见变化。说明直接读取冻结输入，不根据生成结果倒推；原始参数仍可展开，结构化结果同时保存 `counterfactual_task`。本次仅改报告，0新模型调用；3项针对性测试通过（含全部24case描述核对），浏览器目视核验通过。
- CPU FasterRCNN读出9帧，AI审阅固定5帧(0/15/33/51/69)。AI可评分成对分母A/P/E/O/T/OP=11/24/18/11/1/24；其他弃权/N/A。P仅可见形状/遮挡代理，T唯一可评分case仍是2D，不称米制ADE。所有人工verdict为null。
- 15项相关CPU测试通过；报告72个视频全解码、原始来源/窗口/分辨率/帧率、249个内部链接检查通过。浏览器检查三列视频及评分表；窄屏改为纵向展示，宽屏保留三列。

完整HTML（含视频）在 `/root/autodl-tmp/runs/worldsim_v75/WS-V75-DOWNSTREAM-FULL-01/20260923-r1/reports/omnidreams/index.html`；本地交付 `outputs/cfbench-20260923/omnidreams/index.html`。Git只保存轻量[摘要](summary.json)、[AI初评](ai-preliminary.json)、[规则](scoring-rules-v1.json)；视频/原始日志/大条件数组保留在run，不在Git重复存储。

## 问题发现，不夸大为结论

正例：LATERAL-ACTOR-02可见单车横移，真实/FACT/CF后段各7/7匹配，中心误差中位数约4.84/2.16/2.14px。负例候选：REMOVE-01挂车残留、REMOVE-05白车残留；LATERAL-ACTOR-03原位置和新位置同时出现骑车人。背景成对高分可能仅因没有执行编辑，不可代替A/O。

事后输入审计发现SPEED-EGO-02干预终点仅差0.000053米，A/O/T弃权；INSERT-02/04/06中心不在drivable_area，LATERAL-ACTOR-02部分不在内。自行车/行人另有路权问题，中心点审计不等于车身、车道、初始碰撞或完整道路合法性。本轮不换case；预存manual_pending资格未被冒充通过。

评测器保存已匹配检测，未保留全部未匹配检测。REMOVE反事实中无目标track使匹配数结构性为0，绝不能计算为删除成功。部分遮挡/小目标/COCO类别不匹配会导致读出弃权，不记生成失败。AI只看固定静帧，不声称已审阅连续视频。开环O只看生成响应/占据变化，未运行AD、碰撞/TTC或闭环安全。

## ReSim与剩余方法的恢复入口

ReSim任务在run下 `resim/queue-result.json`，每case有分支日志与结果。正在使用公开普通checkpoint而非EMA。用户授权后先测试phase offload-only仍OOM，再用实现原有17帧VAE分块路径完成工程测试；此变体改变时序边界，非原生49帧数值等价。输入为事件前9张RGB，未来40位置仅重复最后历史图占位，ego waypoints延伸到4s；输出49帧中4–27对齐公共窗口。相对其他方法额外4帧历史披露，未来RGB不泄漏。6ego可运行、18actor不支持，报告保留24行但不生成假结果。`scripts/run_resim_cfbench_queue.py`单GPU串行，任一分支失败停止。

HUGSIM预处理输入已经位于 `/root/autodl-tmp/data/worldsim_v75_downstream_bench/reconstruction_inputs/hugsim/{scene-0230,scene-0242,scene-0255}`，各196帧×6相机；actor轴置换与八角点等价断言通过，保留所有case目标含静止车辆。LiDAR地面高度约1.477/1.403/1.454m。适配10Hz而非官方ASAP12Hz，原生idx%30>=24等价源帧%5==4 holdout。还需官方InverseForm语义、UniDepth深度、点云merge、train_ground、train/export；不能把RGB准备叫重建完成。

HUGSIM环境 `/root/autodl-tmp/envs/hugsim-impact` 新增open3d0.18/runx0.0.6及其小依赖，未替换Torch/Numpy。官方InverseForm HRNet48权重289390607字节、UniDepth模型1415383604字节均已完整下载；UniDepth官方fork位于external/worldsim_v75_downstream_bench/UniDepth-HUGSIM，权重下载日志 `hugsim-unidepth-ranges.log` 显示169/169 range完成，不用稀疏预分配大小冒充完成。尚未安装UniDepth/apex或运行GPU预处理。

StreetGS旧实现为 `/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0` 的DriveStudio适配版，env `drivestudio` Python3.9/Torch2.1.2cu118/gsplat1.3，CUDA11.8。MultiTrainer CPU导入通过。勿复用旧wrapper中的旧数据/哈希逻辑；新增配置直接调用tools/train.py，核验三场景/相同holdout和停车目标可编辑后再训练。原数据loader已有用户/历史补丁，不覆盖。还没有新checkpoint。

GaussianDWM公开715个Gaussian帧并非三个完整对齐scene；仍需真实RGB/Gaussian/CLIP特征、paired状态对齐或对齐重建导出。此前合成QA只算smoke。DriveEditor跳过，未把资源不满足记为质量失败。

## 复现与脚本

`prepare_cfbench_maps.py`、`prepare_omnidreams_cfbench.py`、`run_omnidreams_cfbench.py`为Omni条件/推理；`prepare_resim_cfbench.py`、`run_resim_lowmem.py`、`run_resim_cfbench_queue.py`为ReSim适配/串行；`evaluate_omnidreams_cfbench.py`为CPU读出；`audit_cfbench_input_effects.py`为事后输入审计；`add_cfbench_original_videos.py`为原始参考；`build_cfbench_html.py`和`verify_cfbench_report.py`为报告/核验。评分规则在生成开始后、输出初审前冻结；后续审计不冒充预注册门。

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=3 /root/autodl-tmp/envs/motionproj/bin/python -m pytest -q tests/test_cfbench_full.py tests/test_cfbench_gpu_smoke.py tests/test_worldsim_v75_downstream_bench.py
/root/autodl-tmp/envs/motionproj/bin/python scripts/verify_cfbench_report.py --report /root/autodl-tmp/runs/worldsim_v75/WS-V75-DOWNSTREAM-FULL-01/20260923-r1/reports/omnidreams
```

参考[What-If World](https://arxiv.org/html/2605.27589v1)的单变量配对与A/P/E/O；其严格二元分数与本地0–4序数诊断不同。本地T/OP为扩展，不冒充原论文评分复现。
