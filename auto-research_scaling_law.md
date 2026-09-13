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

3. **先确认 failure 值得解决，不要求先有漂亮主方法。**
   先用当前任务中的失败发现和简单强控制确认技术缺口，再选择优化办法。不要因为尚无宏大的 representation/operator 故事，就拒绝有实际效果的简单解；也不把小技巧自动宣称为主贡献。

4. **Failure-first，方法和创新表述由证据决定。**
   先找到真实技术缺口，再选择自然的数学工具。禁止先找最优传输、因果、对偶或变分等术语，再硬套问题。创新比较保留，但不以预设 fancy mechanism 的必要性作为研究目标。

5. **方法是否有效优先于它叫什么。**
   表征、生成过程、查询算子、优化状态、更新规则和信息流都可以成为解法；不强制每次先发明新对象。主方法的贡献范围由任务改善及强控制决定，普通求解或简单优化有效时先承认并使用它。

6. **优化器服务于方法，不能替代方法。**
   可以使用混合整数规划、二次规划、线性规划、增广拉格朗日等成熟求解器做强控制。
   但如果成熟求解器已经解释全部收益，那么“专用新求解器”不再具备主要创新价值。

7. **强控制优先于弱基线。**
   每个创新都要问：

   > 最简单、最成熟、最强的已有办法能不能完成同样的事情？
   > 如果答案是能，就不能靠和旧弱基线比较建立 novelty。

8. **先站住一个有效主方法，再看次要方法。**
   “A + B + C” 不自动等于贡献。主方法已经有效时就整理 problem–method–evidence 的 paper story，不等多个模块齐全；后加方法只针对实际剩余 failure，并交代独立增量，不能靠组合掩盖主方法无效。

9. **同一阶段集中解决一个主要问题。**
   最多保留一个有依据的 fallback，不同时追新 backbone、loss、topology、renderer 等多条路线。找到有效主方法后，可以继续寻找服务同一目标的次要方法，但不把它们当成主方法成立的先决条件。

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

27. **Failure discovery 要导向有效解，不能无限自我解释。**
    诊断用于找出可优化的问题并帮助选择方法；不能先定机制，再反复挖理由维护它。没有有效主方法时，分析越来越精细也不自动提高论文准备度；主方法已经有效时，应及时围绕现有证据写 paper story，不等待所有机理都解释完。

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

31. **Scaling 的目标是积累可复用的失败知识与有效解。**
    抽象升级应由尚未解释的 failure 驱动，不把 loss → operator → representation → generative process 当作必须逐级攀升的路线。低层普通方法已经有效时先使用；只有当前证据确实需要时，才升级解释或计算对象。

32. **论文围绕清楚的问题、有效主方法与相称证据组织。**
    不强制每篇先有“一个新对象 + 一个新机制”。先找到能 work 的主方法，就可以形成 paper story；与现有方法的关系、实际增量及尚未完成验证如实写清。后续次要方法有用才加入，不为十个模块或大而全系统延长探索。

33. **禁止 mechanism-first 的必要性追认：failure discovery → 优化 → 主方法 → story → 次要方法。**
    2026-09-13 用户新增；本条优先解释前述原则及旧研究计划。

    - 禁止先有 fancy mechanism，再不断寻找 failure、理由或改写问题来证明它必要；研究目标是解决实际 failure，而不是维护预设机制。
    - 先做好 failure discovery：说明实际任务中哪里失效、已有证据是什么、简单强控制能解释和解决多少。随后选择最自然的优化办法，允许简单方法成为有效解，不增设繁琐门控。
    - 但凡找到一个确实能 work 的主 method，就立即以已有证据组织 paper story；不等待完整理论、完美 novelty 措辞或配套模块凑齐。这里的 work 指目标任务上的实际有效性，不能用单一局部指标掩盖关键退化。
    - 主方法站住之后，再寻找针对剩余 failure、同样能 work 的次要方法；没有增量就不加，不把次要模块当作补救无效主方法的装饰。
    - paper story 是清楚表达“什么问题、如何改善、证据支持到哪里”，不补写未做实验、不隐藏强控制、不将事后叙事伪装成预先成立的机制必要性。

    V7.4 的教训：诊断更精细不等于主方法更有效；后续研究从已保存的 failure 与可行改进出发，不因某个机制名字有吸引力就继续投入。
