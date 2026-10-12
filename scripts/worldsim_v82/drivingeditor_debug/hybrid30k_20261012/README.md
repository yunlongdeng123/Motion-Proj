# 2024继承目标与30k消融

按用户最新规则保留2024活跃机制，原始动态alpha完整MSE、静态RGB、寿命/速度及深度衰减为共同基础；只切换corrected SGCM和原公式NPS。每臂从零30k，10k/20k/30k完整检查点；C00/G10/N01/GN11共120k步。

```mermaid
flowchart LR
  I[RGB LiDAR 位姿 body mask] --> B[24继承目标和两分支高斯]
  B --> G[可选SGCM 支撑与clone谱系]
  B --> N[可选NPS 原公式与梯度检查]
  B --> R[四臂30k]
  G --> R
  N --> R
  R --> O[完整检查点 分层与静态时间控制]
```

远端准备根目录为`/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012/`，不覆盖原S/V/作者代码。CPU准备不导入renderer、不加载模型；第三方完整源码留在该远端checkout，本Git目录只放工具、局部helper和策略。

默认只检查：

```bash
/root/autodl-tmp/envs/drivingeditor/bin/python -B tools/run_matrix.py
```

用户开GPU后由当前聊天执行：

```bash
/root/autodl-tmp/envs/drivingeditor/bin/python -u tools/run_matrix.py --run
```

GPU计算由同一`gpu_slot.lock`独占，四臂worker串行运行，失败不自动重跑/改配置/删旧资产。GPU环境沿用编译rasterizer对应的DrivingEditor Python3.9；CPU源码生成使用motionproj Python3.10。现有raw-body mask缺shadow，49帧属于开发重建视图；真实GPU首步、质量及作者完整预处理仍未验收。

完整机制继承表、预算、输入角色、组件图、SGCM区域测量和证据见[实验协议](../../../../docs/v82/DRIVINGEDITOR_HYBRID30K_PLAN.md)。
