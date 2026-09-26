# 独立失败卡

由 `scripts/build_research_failure_index.py` 生成；卡片记录各次证据与结论，不是当前研究状态。[旧版本](VERSIONS.md) · [全部 ID](IDS.md) · [维护方式](README.md)。

| ID | 证据主题 |
|---|---|
| [V74-F01](entries/V74-F01.md) | 新的 nuScenes FINAL 身份缺口 |
| [V74-F02](entries/V74-F02.md) | H1 P0 无卡资源与后续恢复 |
| [V74-F03](entries/V74-F03.md) | WEX 自由约束关联缺失 |
| [V74-F04](entries/V74-F04.md) | WEX 未建立独立方法增量 |
| [V74-F05](entries/V74-F05.md) | DCS 生片锚点重复调度 |
| [V74-F06](entries/V74-F06.md) | NKSR 官方环境与编译 |
| [V74-F07](entries/V74-F07.md) | WEX 最近候选责任排序 |
| [V74-F08](entries/V74-F08.md) | RIF 显式松弛求解成本 |
| [V74-F09](entries/V74-F09.md) | DCS 真实联合收益失败，合成正例保留 |
| [V74-F10](entries/V74-F10.md) | RIF 覆盖增加伴随留出提前命中 |
| [V74-F11](entries/V74-F11.md) | NKSR 空层导出失败 |
| [V74-H2-F01](entries/V74-H2-F01.md) | FIT 时间块并非每个对象都有正输入和正监督 |
| [V74-H2-F02](entries/V74-H2-F02.md) | 低配 CPU 的 AV2 转换迁移与相对路径错误 |
| [V74-H2-F03](entries/V74-H2-F03.md) | 亚 float32 分辨率的边界外点会被舍入到闭边上 |
| [V74-H2-F04](entries/V74-H2-F04.md) | 未合并重复顶点导致 NKSR 预算简化提前停滞 |
| [V74-H2-F05](entries/V74-H2-F05.md) | 截断深度破坏远层顺序和完整特征 |
| [V74-H2-F06](entries/V74-H2-F06.md) | 合成种子加编号造成角色间构型重合 |
| [V74-H2-F07](entries/V74-H2-F07.md) | GPU教师初始接口与候选描述实现修正 |
| [V74-H2-F08](entries/V74-H2-F08.md) | 教师轨迹拟合与自身几何闭环出现偏移 |
| [V74-H2-F09](entries/V74-H2-F09.md) | 稳定归属下的首面连续前移，而非薄片边界换面 |
| [V74-H2-F10](entries/V74-H2-F10.md) | 普通法向校正解释旧退化，但联合几何问题尚未关闭 |
| [V74-H2-F11](entries/V74-H2-F11.md) | V7.4 收尾与机制先行的规划纠偏 |
| [V74-H2-F12](entries/V74-H2-F12.md) | 旧适配融合证据不能替代官方VGGT后继的共同失效 |
| [V74-H2-F13](entries/V74-H2-F13.md) | 共享早交不等于普遍幻几何；表面覆盖与物理首回波仍有缺口 |
| [V74-H2-F14](entries/V74-H2-F14.md) | 首回波不进入LTF；碰撞采样改变闭环，但自然前馈危害未确认 |
| [V74-H2-F15](entries/V74-H2-F15.md) | 早交改变LiDAR与车辆轨迹，但初批未证明严重危害；深度读出需归一化 |
| [V74-H2-F16](entries/V74-H2-F16.md) | 近车六日志的唯一新增接触主要受观测范围支配，不能立普遍phantom危害 |
| [V74-H2-F17](entries/V74-H2-F17.md) | 空间回波恢复不等于局部几何因果，普通右前路面修复未恢复残余 |
| [V74-H2-F18](entries/V74-H2-F18.md) | 闭环轨迹偏差不等于几何危害，8k试跑主要差距经RGB传递 |
| [V74-H2-F19](entries/V74-H2-F19.md) | 稳定仿真感知退化也不能自动归为几何，强度恢复提供替代解释 |
| [V74-H2-F20](entries/V74-H2-F20.md) | 局部几何能影响感知，但删除控制已恢复，距离改善也不保证检测改善 |
| [V74-H2-F21](entries/V74-H2-F21.md) | 跨日志框精度下降不等于前车严重丢失或驾驶仿真危害 |
| [V74-H2-F22](entries/V74-H2-F22.md) | 新来源的真实驾驶基线未达到几何归因条件 |
| [V75-F01](entries/V75-F01.md) | 官方 batch 命令未绑定逐案例图像与条件输入 |
| [V75-F02](entries/V75-F02.md) | DriveEditor 原生对象编辑与 10 秒全 case 评价的能力边界 |
| [V76-F01](entries/V76-F01.md) | COLMAP 无序 image ID 导致 VAD-GS 初始化 track 绑定错误 |
| [V76-F02](entries/V76-F02.md) | 同场景 SAM 框提示把遮挡物绑定为隐藏行人 |
| [V76-F03](entries/V76-F03.md) | 当前 HUGSIM / VAD-GS 资产的对象级质量不足以支撑高保真反事实编辑 |
| [V77-F01](entries/V77-F01.md) | 冻结 Ω 的框内点集尚不能直接当作完整可编辑对象 |
