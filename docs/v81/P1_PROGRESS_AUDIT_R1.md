# P1 短周期审计：先确认训练与采样，再进入长预算

task/run：`WS-V81-SEEN-TO-SCENE-P1-20261009/r1`，2026-10-09。用户要求查看当前进度、数据构造并与原文比较，避免七天结束后才发现基础问题。两名独立 `gpt-6-sol / xhigh` subagent 分别只读审计训练与推理；没有使用 fast，也没有人工准入。

```mermaid
flowchart LR
    X[25帧真实RGB + 外绘mask] --> C[RAFT / VAE / CLIP]
    C --> F[光流补全 + 潜变量传播]
    F --> U[SVD时序层]
    U --> V[固定valid视频]
    V --> Q[subagent质量门]
    Q -->|可信且开始恢复结构| T[下一小段训练]
    Q -->|工程或协议疑点| H[保留断点并诊断]
```

## 当前实测与输入

23:31 新加坡时间快照：739/100,000 更新，最近100步均值5.823秒，按同速剩余纯训练6.69天；该值不包含保存、验证，也不保证画质。数值和三个组件的梯度有限，冻结参数无梯度。最新完成生成的断点仍是step2；独立审核为饱和色块和明显结构/时序失真，不是可用视频。两步无法判定长训最终失败。

train 包3471视频、94588张JPEG；1951视频严格>25帧，19313连续窗口。100K是虚拟采样/优化预算，不是独立视频数。每次均匀选视频、随机连续25帧，缩放中心裁剪256²，左右各84像素洞；监督始终是真实视频。没有伪标签、SAM车辆贴片或P2驾驶数据。CPU审核页按真实日志复现step1、2、300，保存源文件名、原图尺寸、监督/可见视频、mask联系图和运行快照。

输入/视频输出在仓库外：`/root/autodl-tmp/reviews/worldsim_v81/WS-V81-SEEN-TO-SCENE-P1-20261009/progress-preview`；同步至本地 `outputs/v81-paper-p1`。审核脚本为 `scripts/worldsim_v81/build_p1_progress_review.py`。播放fps7仅为导出/模型条件，不据此宣称数据原始帧率。

补充独立目视QA：三个输入联系图的五张抽帧均有合理姿态/镜头变化，未见明显重复、错序或损坏；可见带、mask和监督角色正确，但五帧不能证明整段时序通过。源JPEG文件名实际每5递增，所以包内连续25张不等于原视频逐帧；正式YouTube评测另用全帧包，作者实际训练帧采样率仍需核对。[数据集官方说明](https://youtube-vos.org/dataset/vos/)亦区分带标注训练帧和版本统计。完整CPU检查41 passed，审核10视频全部解码、23本地链接无缺失。

## 原文与公开代码的实质边界

来源：[论文§4、§5.1、附录D.1](https://arxiv.org/html/2604.14648)、[固定公开代码](https://github.com/InSeokJeon/Seen_to_Scene/tree/2a9dfc9888e44c7fd00b08af41ef967ae46b6323)。

| 项目 | 已核对事实 | 本轮判断 |
|---|---|---|
| 训练采样 / mask | P1基本对齐公开dataset.py与util.py，25帧、256²、双侧84px | 基本一致；不是当前驾驶DELETE训练 |
| 训练传播 | P1 `train_p1.py:78`未传flow_pairs_info；固定latent_warping.py:188–194因此走All Frames；真实日志同样打印All Frames | 与论文m=4参考链有实质差距，不能称严格论文等价 |
| 官方训练可执行性 | train.py:438传入传播模块不接受的orig_lats，固定签名line170无此参数 | P1删掉该无效参数可运行，但这不解决上述传播差距 |
| GT与条件 | 完整RGB算RAFT和首帧CLIP，masked RGB进入条件VAE；QUERY仅可见RGB | 对齐公开train.py，但训练/QUERY有条件分布风险 |
| 参数与损失 | FCNet、潜变量传播、SVD temporal transformer更新；空间层、RAFT/VAE/CLIP冻结；diffusion+flow L1+ternary warp | 主要对齐公开训练；不是只训练新Adapter |
| 优化与硬件 | 论文Adam、两张A6000；源码AdamW、per_gpu_batch_size=1；当前单3090、bf16、CPU卸载 | 明示差异；作者双卡启动方式/有效batch未公开，不能断言完全一致 |
| 推理 | 论文Gaussian初始噪声前向生成；公开test.py:611另执行inverse | 主literal-public继承源码；单独feedforward只作诊断 |

公开inverse还有可证实风险：test.py:473把B1复制B2，483–484广播更新后两支latent不同，639–647再作为CFG两支，653–655只解码第一支。反演把UNet输出当epsilon使用，而SVD scheduler是v_prediction，且alpha索引不是实际scheduler timestep。静态审计足以要求短对照，不能证明它是霓虹色块唯一成因。训练EDM、VAE目标/条件标度、解码缩放未发现本仓库独有的确定错配。

现成literal/feedforward两路径还存在RAFT洞区-1/0填值及RNG消费差异，不能把两者画质差直接作为“反演导致失败”的因果证据。若分叉，下一次必须固定同一CLIP/flow/传播条件和同一初始latent，仅切采样分支，先处理协议差距，不加P2/P3创新。

## 已实施的节省预算措施

控制器新增1000、5000步助手质量门：各做预选3个valid×两倍率；1000另做同断点首个valid/.33的feedforward诊断，输出独立目录。随后只接受对应`quality_gates/stepXXXXXX.json`、`reviewer=assistant`、`decision=continue`；缺失或hold不自动启动后续训练。8项流程测试通过。

部署采用控制器交接，未中断GPU训练PID29673：保存原控制器状态与命令后，仅替换旧父控制器PID29301。交接进程33980等待当前1000步训练退出及完整断点，再启动新控制器；不会并行加载GPU模型。证据在run的`checkpoint_handoff.json`和`controller_state.before_quality_gate.json`。不等待用户审核。

质量门查看原生与硬合成的完整25帧，比较固定输入的结构、边界和时序，不把训练损失下降当画质证据。若仍为色块、存在采样/数值错误或论文协议未澄清，保留断点做有界诊断；不能无条件放行七天训练。只有生成开始恢复结构且工程链路可信，才考虑到5000步。正式150个测试ID不用于调参。

正式四指标尚未计算，`protocol_verified=false`保持；failure_ledger_refs=[V77-F02]、failure_ledger_delta=none。本轮发现是公开复现工程/协议风险，尚未构成方法失败。媒体、数据、第三方源码、权重不入Git，源码ZIP仍受100MB上限约束。
