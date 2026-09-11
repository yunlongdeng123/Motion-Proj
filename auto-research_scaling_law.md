## Auto Research Scaling Law Rules

1. **问题连续，不轻易换赛道。**
   每个新版本必须建立在上一版本已经确认的失败边界上继续推进，而不是一失败就换成另一个完全不同的问题。
   目标是形成：

   $$
   V_n\text{ failure}\rightarrow V_{n+1}\text{ new abstraction}
   $$

   而不是：

   $$
   V_n\text{ failure}\rightarrow \text{random new topic}.
   $$

2. **优先升级抽象层级，不优先加补丁。**
   如果同一种失败在多种损失、优化器、表征下重复出现，就不继续调权重、阈值、种子，而要怀疑更高层：

   * 任务定义；
   * 表征范式；
   * 生成机制；
   * 几何算子；
   * 优化动力学。

3. **先问“如果它有效，够不够成为论文主方法？”**
   一个候选在写代码前必须先过这一关。
   如果即使实验成功，最后只能写成：

   > 增加一个损失项 / 置信度头 / 后处理过滤 / 超参数技巧
   > 那通常不值得投入主实验资源。

4. **创新优先（Novelty-first），但不是数学名词优先。**
   不是先找：

   * 最优传输；
   * 因果推断；
   * 对偶优化；
   * 变分法；
     再硬套问题。
     而是先找到真实技术缺口，再选择最自然的数学工具。

5. **新方法必须改变一个核心计算对象。**
   候选最好至少改变下列之一：

   * 表征（representation）；
   * 生成过程；
   * 查询算子；
   * 优化状态；
   * 几何更新规则；
   * 信息流。
     只改变损失系数通常不够。

6. **优化器服务于方法，不能替代方法。**
   可以使用混合整数规划、二次规划、线性规划、增广拉格朗日等成熟求解器做强控制。
   但如果成熟求解器已经解释全部收益，那么“专用新求解器”不再具备主要创新价值。

7. **强控制优先于弱基线。**
   每个创新都要问：

   > 最简单、最成熟、最强的已有办法能不能完成同样的事情？
   > 如果答案是能，就不能靠和旧弱基线比较建立 novelty。

8. **方法级创新和系统组合严格区分。**
   “A + B + C” 不自动等于新方法。
   必须指出：

   $$
   \boxed{\text{新增的不可替代机制是什么}}
   $$

   以及去掉它之后，为什么现有方法做不到同样的事。

9. **一个版本最多一个主科学命题，最多一个 fallback。**
   不再同时追：

   * 新 backbone；
   * 新 loss；
   * 新 topology；
   * 新 renderer；
   * 新 uncertainty；
   * 新 simulation story。
     主命题失败就按预定规则停，不把失败者拼成“大框架”。

10. **候选宁缺毋滥。**
    如果只能想到 3 个真正强的候选，就只做 3 个。
    不为了“并行大逃杀”数量好看，再塞 2 个明显弱的方法浪费资源。

11. **候选先过最小证据实验，再进入大训练。**
    每个方法都必须有一个能快速区分：

    > 核心机制到底存在不存在
    > 的最小实验。
    > 如果 1–2 天的小实验已经否定 central hypothesis，就立即关闭。

12. **能当天证伪的事情，不排到第十天。**
    所有可计算的：

    * 理论上界；
    * oracle；
    * representation capacity bound；
    * 简单强控制；
    * 闭式下界；
      必须优先做。
      先判断这条路有没有 headroom，再开发复杂系统。

13. **“失败”必须区分层级。**
    至少分成：

    * 工程失败；
    * 数值失败；
    * 优化失败；
    * 表征失败；
    * 科学假设失败；
    * 新颖性失败；
    * 数据证据不足。
      不能把 solver 没收敛说成方法不行，也不能把工程 bug 修复后自动宣布科学假设成立。

14. **负结果要形成边界，不只是记录日志。**
    每个失败版本最终必须回答：

    > 以后哪些方法族不用再试？
    > 哪个假设被排除了？
    > 哪个假设仍然开放？
    > 这才产生 Auto Research 的累计效应。

15. **已关闭的方法族不换名字重开。**
    例如已经证伪的：

    * 删除式安全；
    * loss weight grid；
    * 简单 opacity 物理化；
    * 已失败的专用 responsibility solver；
      不允许换术语后重新进入主线。

16. **不要把局部指标提升升级成机制成功。**
    例如：

    * coverage ↑；
    * early ↓；
    * recall ↑；
      单独一个好看都不代表方法成立。
      要看任务真正要求的 Pareto 是否同时改善。

17. **如果评价对象是全局算子，就不能只优化局部 proxy。**
    比如我们的最终读出是：

    $$
    Q(S,r)=\min\{t:o+td\in S\},
    $$

    那么只优化点距离、局部表面平滑、单面置信度，都不能自动保证 first-return 正确。
    新方法必须和最终查询语义对齐。

18. **同一失败跨表示重复出现时，升级到更高层解释。**
    如果：

    * Mesh；
    * Open Chart；
    * Implicit Field；
    * Neural Surface；
      都出现类似：

    $$
    coverage\uparrow,\ first\text{-}return\ physics\downarrow,
    $$

    就不再把问题归因某个具体网格结构，而要研究更本质的生成语义。

19. **优先解决底层生成问题，不用 UNKNOWN 逃避重建。**
    选择性查询、不确定性、认证可以作为：

    * 诊断；
    * Verified Bake；
    * asset validity；
      但如果主任务仍是仿真资产生成，就不能把大量 `UNKNOWN` 当成几何重建成功。

20. **Precision-first 应进入生成动力学，而不是只进入后处理。**
    我们可以接受“错误表面比缺失表面更昂贵”的任务偏好，但应该体现为：

    * surface birth；
    * support extent；
    * split / merge；
    * primitive growth；
    * first-return competition；
      等生成机制，而不是仅仅给 early loss 乘更大权重。

21. **视觉大模型作为先验，不默认作为最终物理几何。**
    视觉 foundation 更适合提供：

    * shape prior；
    * latent geometry；
    * correspondence；
    * completion prior。
      稀疏 LiDAR / ray evidence 决定：
    * metric alignment；
    * visible support；
    * free-space；
    * first-return physics。

22. **GPU / CUDA / BVH 是实现加速，不是创新本身。**
    除非表示和算法正因为硬件查询特性而共同设计，否则“用了 GPU 光追”不算主贡献。

23. **外部 SOTA 用来定义问题边界，不只是刷数字。**
    如果 NKSR reconstruction 很强但 first-return 仍差，这说明：

    $$
    \text{reconstruction quality}\neq\text{ray-query quality}.
    $$

    这类结果应服务于 problem statement，而不是简单宣称“我们打败 NKSR”。

24. **Claim–Evidence Matching。**
    claim 多大，证据就要多大。

    * 只声称 nuScenes 域内有效，可以单数据集；
    * 声称跨 sensor / 跨域，就必须跨域验证；
    * 声称通用 representation，就必须更广泛测试。
      不强制每篇论文都多数据集，但不允许超证据扩 claim。

25. **开发集和最终确认集角色绝不能混。**
    一旦某个数据集已经参与：

    * 方法选择；
    * 阈值选择；
    * failure diagnosis；
      就不再称 independent test。
      后续只能作为开发资源。

26. **不要把统计严谨误当方法贡献。**
    bootstrap、置信区间、校准、风险控制很重要，但：

    > 严谨 ≠ 创新。
    > 统计工具用于防止错误结论，不用于弥补 method novelty。

27. **不要把 Failure Analysis 本身当主论文。**
    Failure analysis 的职责是：

    $$
    \text{定位下一层创新空间}.
    $$

    如果只把失败解释得越来越漂亮，却没有新的核心机制，论文仍然是弱拒。

28. **每轮都要有明确的停止规则。**
    失败后允许：

    * 修明确工程 bug；
    * 补一个预注册 fallback。
      不允许：
    * 无限 seed；
    * threshold grid；
    * loss grid；
    * 临时换 backbone；
    * 临时加模块救结果。

29. **失败资产必须完整保留。**
    至少保存：

    * failure case；
    * 关键中间状态；
    * first degradation；
    * responsible rays / faces / primitives；
    * solver state；
    * 最强控制结果；
    * 正结果与负结果。
      这样下一版本可以真正继承，而不是重新猜原因。

30. **每个版本最终必须形成一句“新知识”。**
    例如目前这条链：

    $$
    \text{V7.2: 小 head 不足}
    $$

    $$
    \text{V7.3: Euclidean coverage}\neq\text{first-return correctness}
    $$

    $$
    \text{V7.4 上半场: 更强求解/增密}\neq\text{联合物理改善}
    $$

    下一版本必须继续沿这个方向增加一条，而不是重新从零开始。

31. **Auto Research 的 Scaling Law 最终目标不是更多实验，而是更高层的抽象。**
    理想 progression 是：

    $$
    \text{loss}
    \rightarrow
    \text{operator}
    \rightarrow
    \text{representation}
    \rightarrow
    \text{generative process}
    \rightarrow
    \text{task formulation}
    $$

    只有当低层已经被证伪，才向上一层升级。

32. **顶会气质最终来自“一个新对象 + 一个新机制 + 一个强证据链”。**
    最理想的方法结构是：

    * 一个清楚的新几何对象；
    * 一个不能被普通强控制替代的新更新机制；
    * 一个直接命中核心 failure 的实验结果。
      不需要十个模块，也不需要大而全系统。