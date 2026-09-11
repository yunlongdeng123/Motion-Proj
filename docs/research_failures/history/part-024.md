# 历史原始记录 024

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V7.1 S1 outcome note — 连续位移形成hazard/surface Pareto（2026-09-03）

- canonical=`run://worldsim_v71/WS-V71-S1-DISPLACEMENT-ORACLE-01/
  20260903T101000Z__s1-displacement-oracle-r1`；64 Actors，oracle-check rays=`36,101`；
- D1：hazard literal early相对下降=`5.802%>=5%`，Chamfer delta=`-3.778mm<=0.5mm`，Actor/hazard retention=
  `100/100%`；candidate displacement action space feasible；
- 与V7-F24--F29区别：所有COMPLETE candidate数量保持不变，仅学习`Delta d_ray/Delta d_normal`；KEEP与matched PROJECT
  为冻结anchor，因此不是删除/UNKNOWN/threshold/router路线的变体；
- remaining boundary：clear literal early相对恶化`4.009%`，M0必须在source selection同时报告hazard/clear/all，不能把D1
  的hazard门升级为全stratum保证；
- decision：执行S2与M0，跳过S1-B；M1只在M0无法逼近该oracle且证据表明candidate support受限时解锁；
- failure_ledger_delta=`none`，下一V7.1 failure ID仍为`V71-F02`。

## WorldSim V7.1 failure ledger 启动（2026-09-03）

### V71-F01 — 独立 V7.1 runner 未自举仓库根目录

- 分类/状态：engineering / resolved；发生于`freeze_worldsim_v71_source_split.py`首次入口，数据与quality尚未读取；
- 观察：在repo root直接执行脚本时`ModuleNotFoundError: motion_proj`，与历史`V6-F66`同类；
- 根因：Python把`scripts/`而非仓库根目录放在`sys.path[0]`；新入口遗漏既有防复发合同；
- 修复：source-split与S1 runner都从`Path(__file__).parents[1]`绑定repo root，随后metadata split成功冻结且V7.1定向
  测试`6 passed`；
- 防重复：后续V7.1独立入口沿用同一自举段；不得靠调用者临时`PYTHONPATH`掩盖正式入口缺陷；
- 证据：task=`WS-V71-S0-IMPLEMENTATION-01`，branch=`research/worldsim-v7.1-learned-evidential-surface`，commit=
  本逻辑提交；failure_ledger_delta=`V71-F01 resolved`。下一可用V7.1 failure ID=`V71-F02`。

### V7-F24--V7-F29 migration note — V7.1 改变表面位置而非复开删除族

- V7-F24证明独立点级三态分类在首返回组合下不闭合；V7-F25/F26证明ray-only与hybrid keep/delete都以Chamfer/hit为代价；
- V7-F27/F28关闭Actor router和one-slot veto；V7-F29把删除族边界限定为early单调而非surface Pareto；
- V7.1唯一新假设是连续`Delta d_ray/Delta d_normal`或条件SDF/SCF能移动/生成表面。它不调旧threshold、不改变Actor/
  hazard状态，也不以UNKNOWN删除冒充表面修复；
- S1必须先在oracle-check rays同时验证hazard literal early相对下降`>=5%`与Chamfer delta`<=0.5mm`。若candidate
  displacement不可行，只允许一次sparse field oracle；A/B均失败即停止learned C1。

## Submission-inventory note — current page count overrides historical milestones（2026-09-02）

- safety-boundary reflow把当前supplement从10页收敛为9页，checklist/final audit一度仍写10页；现已同步；
- 早期status/experiment条目中的10页是当时真实构建记录，不应批量改成9页，否则会破坏研究时间线；
- 这是documentation consistency fix，不是scientific failure，不消耗ID；下一可用仍为`V7-F30`。

## Safety-section layout note — deployment boundary kept readable（2026-09-02）

- 首版较长deployment段落跨two-column/one-column float boundary，造成page 4底部孤立续文；在上传前压缩重复措辞并移到
  Actor-level interval之后，使完整安全段落落在page 3、failure table/figure同在page 5；
- 未缩字体、图或表，未删除false-repair/sensor-shift/hazard-coverage风险及production/closed-loop/safety-case禁用边界；
- final supplement为9页且无overfull/undefined引用；这是未发布的writing/layout recovery，不消耗failure ID；
  下一可用仍为`V7-F30`。

## Bibliography-rendering note — source typo fixed before release（2026-09-02）

- `kalble2024accurate`原作者字段为`K{"a}lble`的缺反斜杠变体，且若干method/sensor专名未防BibTeX sentence-case；
- 只修转义并保护正式大小写，未改变引用对象、作者身份、venue、year、pages或scientific text；
- 三个入口重新构建，最终references视觉正确且无undefined引用；这是publication source fix，不消耗failure ID；
  下一可用仍为`V7-F30`。

## Certificate-layout note — formal proof without a hidden broader guarantee（2026-09-02）

- 初版单行`e_Q`公式产生70.33pt overfull，改为aligned两行；Proposition最初跨page 2/3，随后整体移至page 3；
- 两项均在本地提交前发现，final supplement无overfull/undefined引用，未缩字体、间距或删除non-guarantee；
- 命题只能用于`S' subseteq S`的纯删除，不得外推到PROJECT/COMPLETE、collision-free、closed-loop或road safety；
- 这是未发布的writing/layout recovery，0 scientific exposure，不消耗failure ID；下一可用仍为`V7-F30`。

## Literature-layout note — stronger positioning without losing the 8-page boundary（2026-09-02）

- CVF官方UniSim/LiDAR4D/DyNFL只用于界定sensor realism与physical collision semantics，不得借引用扩大HARP-3D claim；
- 首版较长Related Work使四行Conclusion进入page 9；在上传/提交前删除重复措辞并恢复全部正文止于page 8，未缩字体、
  行距、图表或安全边界；
- final page 9仅references；新增三项引用均已解析，原Table 1 `6.03pt` non-clipping overfull不变；
- 这是未发布的layout recovery，0 scientific exposure/experiment，不消耗failure ID；下一可用仍为`V7-F30`。

## arXiv identity-boundary note — shared science, isolated author metadata（2026-09-02）

- conference与arXiv入口共享同一title/body/sections/tables/figures/macros/bibliography，禁止分别复制后产生claim漂移；
- 真实作者块只能进入Git忽略的`paper/arxiv_author_metadata.tex`，不得写入anonymous `main.tex`或commit history；
- 本地临时匿名metadata只用于证明新入口可编译，不上传、不称为non-anonymous arXiv deliverable、不发布dummy PDF；
- 缺真实authors/affiliations/category/license不是scientific failure，也不授权继续读test或改结果；下一ID=`V7-F30`。

## Final-audit boundary — science complete does not fabricate publication metadata（2026-09-02）

- 29/29 paper registry runs与P1/P2落盘，60/60 AV2 cohort目录、30 panels/30 videos存在；未发现需要恢复的科学资产；
- 原计划五类visual role被明确合并为2张main figures+6张main tables，完整曲线留在supplement；这是8页布局适配，
  不能写成字面“主图1--5均独立存在”；
- non-anonymous arXiv和final CVPR2027 package仍缺真实作者/单位/联系信息、category/license/acknowledgement、paper ID及
  尚未发布的official kit；禁止虚构或以继续读test来填补等待；
- 终局审计不新增训练、数据读取、门或scientific verdict，不消耗failure ID；下一可用仍为`V7-F30`。

## Figure-convergence note — hard evidence promoted without claim promotion（2026-09-02）

- main Figure 2把literal correction、set-deletion direction/utility cost和sensor-opportunity sensitivity并列，不能据此升级
  为collision、closed-loop、policy、domain-invariant或road-safety guarantee；
- 完整P7 coverage curves保留在supplement，主文裁剪只改变版式，不改变threshold、point、metric或结果；
- 首次版式把PNG pixel误作TeX point导致右面板不可见；视觉检查后改用真实自然尺寸的`700bp`裁剪，未提交坏PDF；
- Conclusion一度跨入page 9，随后通过删重复文字恢复references-only page，不缩字体/间距；
- 均为提交前layout repair，不是scientific failure，不消耗ID；下一可用仍为`V7-F30`。

## Submission-boundary note — external metadata is not a research failure（2026-09-02）

- official CVPR2027 kit尚未发布；保持CVPR2026 official style的provisional匿名稿，不使用第三方模板或猜测2027规则；
- 作者顺序/单位/联系信息、arXiv category/license/acknowledgement与conference paper ID必须来自真实外部输入，禁止虚构；
- 等待这些字段时不得复开test cohort、训练、threshold sweep或method branch来“继续优化”已冻结结论；
- anonymous与arXiv版本必须共享同一sections/tables/figures/result macros，只允许身份、发布模式和届时official style差异；
- 这不是scientific failure，不消耗ID；下一可用仍为`V7-F30`。

## P23 paper claim audit — fresh measurement transfer kept separate from model transfer（2026-09-02）

- 摘要/正文/补充材料都以P23 `14.817%`/`8.29x`作为fresh metric confirmation，同时把P22保留为consumed correction；
- AV2-C/AV2-F在图中显式区分；P3/P15继续只拥有target-nearest proxy，P20/P22/P23拥有literal measurements，
  P21只拥有deletion theorem；
- 禁止把fresh measurement/provenance复现表述为P16/P4 model generalization、policy validation、collision或road safety；
- 8页正文与引用/视觉检查通过；没有为排版删除failure、cohort、负结果或non-guarantee；
- P23完成后关闭新AV2 cohort与first-return方法扩展；无新failure，下一ID=`V7-F30`。

## P23 outcome note — fresh cohort confirms the first-return metric failure（2026-09-02）

- fresh all/hazard/clear literal exposure=`10.801/14.817/6.750%`，分别为proxy的`7.89/8.29/7.12x`；预冻结两门全过；
- 70,220 literal new-early全部COMPLETE；fresh hit/early=`1.237`复现P22 consumed `1.194`附近的低效率，而非proxy高比；
- 该正结果只升级“target-nearest严重低估literal first return”与“COMPLETE拥有新提前终止”的证据等级；不升级
  P16/P17/P17R/P19或P4 selector，不把measurement transfer称为model/policy transfer；
- downloader与唯一P23进程正常退出，10 logs无retry；不重跑、不扫tolerance、不再建立新AV2 holdout；
- 无scientific failure，`V7-F30`保持下一可用ID。论文必须以P23 fresh数字作为外域独立确认，并保留P22 consumed纠错。

## P23 prevention note — prospective metric confirmation, not target reuse（2026-09-02）

- 第三批10 logs在P22前已按metadata冻结，P23 freeze时quality/model output仍未读；允许一次新的prospective metric
  confirmation，不回写P22 consumed verdict；
- proxy与literal在同一compile、同一Actor/ray上计算；`.20m` lateral/depth tolerance、P2 surface与PROJECT不可变；
- 只设all/hazard literal rate高于proxy两门；不得按结果换log、删失败log、扫tolerance、增加stratum门或重跑；
- 无论结果都不得选择或promote P16/P17/P17R/P19/selector；只能确认或否定metric undercounting；
- 两门任一失败登记`V7-F30`；全部通过则不消耗failure ID，下一可用仍为`V7-F30`。

## P22 paper correction outcome — proxy claim retired without hiding the metric failure（2026-09-02）

- main/supplement现在逐处区分target-nearest proximity与literal minimum-positive-depth first return；P3/P15原数值保留为
  proxy diagnostic，不再承担first-return、visibility certificate或高hit/early物理效用论证；
- P20/P22拥有source/consumed-AV2 literal measurement，P21只拥有deletion early单调性；图表显式展示source约6倍与
  AV2约7.3倍低估，不把外域已消费纠偏包装成fresh transfer；
- AV2 literal hit/early=`1.194`且142,022 new-early全部来自COMPLETE，原proxy `14.51`叙事已撤回；
- 论文仍为8页正文，第9页纯references；supplement扩为10页，无undefined citation/reference或视觉裁切；
- 这是claim/metric语义修复，不新增scientific failure；禁止用P22选择deletion/tolerance/policy，下一ID仍为`V7-F30`。

## P22 outcome note — AV2 confirms proxy failure; paper claim correction required（2026-09-02）

- canonical P22把AV2 all/hazard/clear new-early从proxy `1.348/1.912/.980%`纠正为literal
  `9.894/13.926/7.256%`，约`7.34/7.29/7.41x`；
- 142,022 literal new-early全部由COMPLETE首占用，new-hit/early=`1.194`，所以P15基于target-nearest的`14.51`
  只能保留为proxy diagnostic，必须从first-return收益叙事撤回；
- 这不是方法失败ID：surface/policy未变，错误在metric语义；P3/P15数值保留但claim ownership收窄，P20/P22拥有literal operator；
- prevention：不以P22已消费AV2选择deletion/threshold/tolerance，不重跑第二cohort，不把约7.3倍写成fresh transfer或
  road-safety bound；下一可用scientific failure仍为`V7-F30`。

## P22 prevention note — external operator correction without target reuse promotion（2026-09-02）

- P22只纠正P15已消费20-log AV2的ray operator；不得把结果包装为第二次fresh test或用来选P17/P17R/P19；
- legacy comparator、Actor identities、compiler surface、PROJECT rule与`.20m` tolerances全部冻结；每个log保留；
- 第三批10-log payload保持unread；P22结果无论方向如何均不触发policy/threshold/tolerance/recovery sweep；
- 若实现入口在metric前失败，只允许同合同窄修复；scientific negative的下一可用id保持`V7-F30`。

## P21 outcome note — exact directional theorem supported, no policy promotion（2026-09-02）

- canonical=`run://worldsim_v7/WS-V7-P21-MONOTONE-SAFETY-BOUNDARY-01/
  20260903T134500Z__monotone-safety-boundary-r1`；
- support：`S' subset S => d_{S'}(r)>=d_S(r)`精确成立，删除不会新增literal first-return early；
- bounded frontier：P19以71 hits和`.0918622mm`换131个hazard events，是冻结三点中最高效率，但仍违反
  non-worsening-Chamfer，故不替换baseline、不授权fresh target read；
- prevention：不把events/hit或events/mm当新objective，不扫deletion capacity/tolerance/threshold，不从定理外推
  collision、policy或road-safety guarantee；
- exposure：只读P20 summary，0 data recompilation/training/refit/target read；`V7-F29`保持准确，下一可用failure
  id=`V7-F30`；
- paper audit：8页正文边界恢复，第9页仅references；supplement新增冻结边界图且仍为9页，无新failure。
- plotting recovery：本地bundled Python缺Matplotlib后未安装新依赖，改用已被P7验证的远端项目环境单进程绘图；
  失败发生在import阶段、0 output/metric/data read，不占科学failure id。

