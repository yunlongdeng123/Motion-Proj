实际长窗 V 的独立 CPU 审阅已通过。审阅绑定的是新 long binding 和真实 B1/B2 的 32000 终点证据；此结论只表示材料可供 root 批准运行，不代表 V 已恢复、效果正常或论文复现通过。

```text
同一原始30k四件状态 + 原467输入
                   ↓
       冻结恢复 observer / 原采样与50k LR
                   ↓
   B1 原版     B2 原版重复     V 原版 + λ=.05 掩码外透明度项
                   ↓
       2000步、相同camera/time_shift/LR
                   ↓
     4个RNG安全节点 → 终点49帧固定域 + 四件完整状态
                   ↓
         预先约定的逐帧效应、原版波动与真实图像判断
```

`final_review_V.json` SHA256：`f9b25211ed63de56ff7a5cf7bf865648a8cfc59be0d3b4bb8c5c33e12a1b5a07`。

- Harness：`93df22a4b8db482af596657ba1d3e86e61a0ad0893dac3af35f85adf872ba1d4`。
- Prepared plan：`1afe9a9b5f05dfcff1d565e064f276ef8161250060eecc0bb2b31a35be031065`。
- Experiment binding：`5701482a50b078e980a39a4aec9b6747cdc1628882b2b48a2dd985d588314260`。
- Guard：`8356201320c757079a5dfaf4cb4968e22bc537fbf8f1dd7e0e6d1b6242f0d42c`。

15 项针对实际材料的检查通过；另在 remote 独立核对 23 个关键小文件的实际 SHA，以及原版和候选各 148 文件的源目录清单。算法仅 train.py、paper_losses.py 两文件变化，与已审阅的短窗候选字节完全一致。完整配置仅路由两字段及显式候选两字段变化；形式恢复 helper、SGCM、原损失、horizon、输入和测量协议均保留。

真实 B1/B2 完整 2000 步、原30k恢复、RNG安全49帧、四件终态的已落盘证据均存在；实际 2000 个 camera/time_shift/LR 完全一致，首十个原损失标量完全一致。后续 CUDA 损失轨迹存在变化，不能把首步相等解释为整条轨迹相等。原版重复的 11 项主测量及 126 项六域上下文测量，其帧均值已独立从真实49帧统计复算；这是一次观察到的重复差异，不是置信区间或通用噪声上界。

V 实际 GPU 恢复和训练尚未执行。运行仍必须证明参数/Adam/RNG/采样器/SGCM 的完整恢复、首步 backward 前十标量相等、全2000步配对、节点评估 RNG 还原、末49帧 GT/域一致以及四件完整状态。审阅不重复大 checkpoint、NPZ 或467数据文件的哈希；这些已由真实 CPU prepare/导出器验证并硬绑定。NPZ 未保存独立 runtime mask 像素或 actualScene pose 快照，这一限制明确保留。

Root 需为这份 exact V 另签 receipt，并交既有 component agent 唯一 GPU；本审阅不启动任何运行。效果判断仍按锁定协议同时考虑 composite、observed non-sky outside actor 静态质量、dynamic alpha/support、目标车辆保真和真实图像/移除局部性，不因 composite PSNR 较高宣布分离正常。
