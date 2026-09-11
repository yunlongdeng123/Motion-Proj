# 历史原始记录 020

> 冻结历史，不是当前执行规则。当前规则见 [scaling law](../../../auto-research_scaling_law.md) 与 [AGENTS](../../../AGENTS.md)。
> 原文件：`docs/RESEARCH_FAILURES.md`；迁移前提交 `abbd431c`。原有相对链接按原目录解释，证据路径原样保留。
> 按索引读取相关小节；旧哈希、过度门控和资源指令均不重新生效。

## V71-F23 — 可学习field补偿了恶化的completed anchors（2026-09-04）

- 分类/状态：auxiliary-decoder compensation / terminal for jointly trainable surface+field；canonical=
  `20260904T150000Z__m19-joint-geometry-return-s71121-r1`；
- 表面判定：预注册5/5通过；Chamfer相对M8=`-2.424mm`，field相对当前point reference的hazard early=
  `+15.556%`、all hit=`+7.131pp`；
- 反事实审计：point hazard early相对baseline由M8的`+5.123%`反转为M19的`-4.102%`；point hit由
  `+2.759pp`降为`+1.887pp`；joint field绝对hazard early=`24.435%`，冻结M18=`24.361%`；
- root cause：surface和field同时可学习时，categorical loss可由query decoder吸收；“field相对当前surface”的门又把
  已恶化surface作为分母，形成合法但不支持anchor物理一致性的补偿解；
- optimization evidence：children平均移动`24.708mm`、frame coverage `-2.472mm`、NLL和depth均下降，无资源或
  数值失败；所以不是没训练，而是可辨识性错误；
- resolution：M19保留为技术gate-pass/科学claim-reject，不启动external、不扫loss/lr/epoch；下一路线移除trainable
  decoder，用anchors+scale直接定义ray categorical energy，使physics gradient无法绕开geometry；
- claim impact：M19支持“联合优化能改善surface coverage和field hit”，不支持“completed anchors本身物理自洽”。

下一可用编号：`V71-F24`。

## V7.1 M19 prevention note — 联合训练必须持续保留原生3D监督（2026-09-04）

- M19允许first-return loss回传到completed children，但不得移除symmetric set、point-to-plane/scale和逐帧coverage；
- field logit只读actor-canonical query/build evidence，不输入target ray direction，避免退化为逐ray记忆；
- observed anchors不可学习，部署保留全部children；禁止UNKNOWN mask、surface filter、threshold或Actor deletion；
- geometry与physics分别报告，只在共享anchors处用PCGrad耦合；trajectory/static/image/hazard不混入；
- 若M8 Chamfer恶化超过1mm或M18式early/hit任一失败，登记`V71-F23`并关闭joint路线，不扫loss权重恢复。

下一可用编号仍为：`V71-F23`。

## V7.1 paper claim prevention note — 监督证据不得退化为后处理叙事（2026-09-04）

- M18改善只能归因于native GT LiDAR first-return one-hot target、全ray categorical proper loss以及同分布部署；
- UNKNOWN mask、surface deletion、调阈值、选择性可见性均不得作为3D physical consistency证据；
- M13--M17负结果必须与M18共同报告，证明local support、任意signed crossing和diffuse density为何不足；
- clear子组负结果与fresh AV2未完成必须保留，禁止使用“全面改善”“安全”或“跨域泛化”措辞；
- image fusion、geometry completion与dynamic/static routing后续必须各自拥有GT/teacher target和独立消融，不能混入同一指标后归因。

下一可用编号仍为：`V71-F23`。

## V7.1 M18 AV2 prevention note — 20/20前不读质量（2026-09-04）

- cohort固定为`configs/worldsim_v71/av2_zero_shot_cohort_v1.json`的20个fresh logs；旧V7 30-log完成标记无效；
- checkpoint固定M18 categorical与M8 point comparator；无fine-tune/calibration/threshold selection/failed-log deletion；
- evaluator允许按`.complete`逐log写`EXTERNAL_ACTORS.partial.jsonl`，但20/20前禁止读取任何metric/row；
- final aggregate同时报告M8 point五门与M18 categorical early/hit两门；任一失败不得删log或回调M18；
- 下载器维持唯一现有进程，不启动第二个；空间不足才暂停external，不删除cohort数据。

下一可用编号仍为：`V71-F23`。

## V7.1 M18 reporting correction — metric路径正确、deployment标签陈旧（2026-09-04）

- canonical M18的ray metrics实际由`_categorical_first_return_partition`生成，使用softmax CDF median；
- generic summary writer仍写入旧`first_positive_to_nonpositive_zero_crossing`字符串，仅为metadata描述错误，不影响计算；
- 已将writer改为读取冻结config的`deployment_description`；不改写完成run、不重跑M18、不把该错误升级为科学failure；
- AV2 evaluator必须显式读取checkpoint `field=categorical_first_return`，不得相信旧summary字符串选择推理路径。

下一可用编号仍为：`V71-F23`。

## V7.1 M18 prevention note — first-return bins必须全ray竞争（2026-09-04）

- 对Actor AABB ray上全部depth bins输出logits并用softmax归一化，总质量固定为1；
- GT return直接量化为唯一one-hot bin，categorical NLL为主目标，expected-depth L1只保留metric sub-bin方向；
- 部署用同一分布CDF的固定中位depth，无opacity threshold、zero crossing、peak filter或事后校准；
- M8 geometry冻结，不加image/trajectory/hazard/ray-drop/dynamic-static变量；
- 若early/hit合同失败登记`V71-F23`并关闭categorical ray head，不扫binning/temperature/loss/threshold/seed/epoch。

下一可用编号仍为：`V71-F23`。

## V71-F22 — 单调density降低early但弥散积累使termination系统性提前（2026-09-04）

- 分类/状态：diffuse volume density / terminal for independent-density accumulation；canonical=
  `20260905T020000Z__m17-ray-survival-s71119-r1`；
- 观察：early all/hazard相对M8下降`22.247/26.520%`、observable=`96.63/96.18%`，但hit下降
  `26.443/26.140pp`，最终6/7；
- 训练审计：loss=`1.387→0.906`、depth L1=`0.754→0.535m`、terminal opacity=`0.584→0.878`、mean
  density=`0.449→0.890/m`；无NaN/OOM；
- root cause：非负density使CDF单调，却允许多个hit前小density积累超过0.5；step-CDF balanced NLL的post-hit项可被提前
  opacity满足，导致正确topology但错误surface localization；
- literature/open-source response：Neural LiDAR Fields补充材料把GT range构造成沿ray的高斯weight distribution并做peak/range
  重建；CaDDN以one-hot LiDAR depth bin监督sharp categorical distribution。迁移为M18全raysoftmax竞争唯一GT bin；
- resolution：关闭独立density accumulation，不调threshold/density/loss；保留monotone CDF边界，改用归一化categorical
  termination mass；
- claim impact：M17只支持“单调表示降低early”的机制证据，不支持surface localization。下一可用ID=`V71-F23`。

下一可用编号：`V71-F23`。

## V7.1 M17 prevention note — first return由单调survival定义（2026-09-04）

- 网络只预测非负metric density；沿有序ray samples计算transmittance与termination CDF，CDF按构造单调；
- GT return监督完整step-CDF与termination depth，不把FREE/hit拆成相互冲突的两个scalar目标；
- 部署首交是同一CDF首次达到固定中位概率`0.5`，不是调优occupancy threshold或删除早期surface；
- M8 point geometry冻结；不加ray-drop、image、trajectory、hazard或dynamic/static混合变量；
- 若early/hit合同失败登记`V71-F22`并关闭当前survival field，不扫CDF阈值/density scale/samples/loss/seed/epoch。

下一可用编号仍为：`V71-F22`。

## V71-F21 — 完整FREE-ray监督保住hit但任意signed field产生多重早交（2026-09-04）

- 分类/状态：ray-order underconstraint / terminal for unconstrained direct signed query field；canonical=
  `20260905T013000Z__m16-full-ray-query-s71118-r1`；
- 观察：hit all/hazard=`+0.642/+2.072pp`、observable=`93.59/93.51%`，但early all/hazard相对M8恶化
  `21.611/11.981%`，最终6/7；
- 机制增量：相较M15，native full-ray queries把missing-surface问题基本消除，证明GT construction有效；失败已转为同一ray上
  arbitrary scalar多次变号/过早首交；
- 优化审计：full-ray FREE=`0.675→0.513`、back occupied=`0.696→0.580`均下降，但hit从`0.0039`升到
  `0.0277m`；后期gradient conflict约`48%`，无NaN/OOM；
- literature/open-source response：NeuRAD以非负opacity和累积transmittance定义LiDAR expected depth；Neural LiDAR Fields用
  physically motivated two-way transmittance与peak/volume rendering得到first return。迁移为M17单调termination CDF，不复制
  ray-drop/intensity支线；
- resolution：关闭arbitrary signed decoder，不增加FREE sample或阈值；保留M16 native full-ray GT target，改成单一survival
  proper objective与同分布部署；
- claim impact：M16支持“native ray supervision恢复coverage”，不支持3D physical consistency claim。下一可用ID=`V71-F22`。

下一可用编号：`V71-F22`。

## V7.1 M16 prevention note — 先补全GT FREE ray再学习query field（2026-09-04）

- 每条target ray的FREE监督覆盖Actor AABB entry到return前方，不再只采`0.10/0.20m`局部front；
- hit zero与`0.05/0.10m` back occupied保持native 3D metric标签，同一scalar直接部署首交；
- decoder必须query-specific读取local M8 children，禁止退化为M3/M4 Actor-global latent；
- 不再内置plane/ball/cylinder primitive support，也不允许UNKNOWN mask、surface filter或阈值校准；
- 若early/hit合同失败登记`V71-F21`并关闭当前direct-query结构，不扫query数/field range/loss/capacity/seed/epoch。

下一可用编号仍为：`V71-F21`。

## V71-F20 — 单侧surface cell改善coverage但有限primitive仍漏失真实表面（2026-09-04）

- 分类/状态：finite primitive coverage / terminal for M10--M15 primitive-support family；canonical=
  `20260905T010000Z__m15-one-sided-cell-s71117-r1`；
- 观察：field early all/hazard相对M8下降`37.425/35.107%`，但hit下降`14.896/13.540pp`，observable仅
  `57.39/60.41%`；最终6/7；
- 对M14的机制增量：observable all/hazard由`37.53/39.89%`升到`57.39/60.41%`，证明GT tangent extent与单侧
  back depth拆分确实扩大覆盖，不是失败方向完全无效；
- 优化审计：hit=`0.694→0.636m`、back occupied=`14.029→12.857`、radius loss约`0.08`；有稳定梯度且无
  NaN/OOM，但有限M8-centered cells仍不能覆盖完整target first-return surface；
- literature/open-source response：QueryOcc官方实现直接沿raw-LiDAR ray采positive/negative 4D queries且不依赖render loss；
  ShelfOcc强调先生成native metric 3D supervision，GaussRender则只把rendering作为3D supervision的辅助项。迁移为M16
  full-ray FREE + hit/back native query监督，不以render或primitive filter补救；
- resolution：关闭M10--M15 primitive support，不扫radius/depth/neighbors/epoch；保留M8点表面正结论，M16只改变query
  field与GT采样合同；
- claim impact：单侧support的机制正证据可作ablation，但不能形成surface coverage或3D consistency claim。下一可用ID=
  `V71-F21`。

下一可用编号：`V71-F21`。

## V7.1 M15 prevention note — surface support必须前后非对称（2026-09-04）

- cell front boundary就是GT hit zero plane，normal support只向behind-return延伸`0.10m`，禁止向FREE侧对称膨胀；
- tangent radius由GT 8NN maximum local extent直接监督，不能复用M8 median scale而不训练；
- actor field仍为所有cells的CSG union，AABB外/所有cells外为FREE；无threshold/filter/mask；
- depth=`0.10m`来自冻结narrow-back supervision终点，不扫描depth/radius multiplier；
- 若early/hit合同失败登记`V71-F20`并关闭one-sided cell，不做第二seed或loss恢复。

下一可用编号仍为：`V71-F20`。

## V71-F19 — compact radial field清除FREE伪面但丢失真实surface coverage（2026-09-04）

- 分类/状态：support geometry coupling / terminal for radial compact patch；canonical=
  `20260905T003000Z__m14-compact-field-s71116-r1`；
- 观察：early all/hazard相对M8下降`70.399/69.115%`，但hit下降`25.333/24.147pp`，observable仅
  `37.53/39.89%`；最终6/7；
- 训练机制：front FREE=`0.0958→0.1112`保持很低，back occupied=`15.513→15.437`几乎不可优化，说明不是
  front false-positive而是compact support覆盖不足；
- root cause：M8 scale由GT 8NN median tangent distance学习，M14却把它作为三维ball radius；ball同时限制切向coverage和
  behind-surface depth，重现M9 scale角色耦合；
- literature/open-source response：Ponder在GT depth附近使用可信near-surface SDF、远处单独FREE正则；QueryOcc强调直接
  query supervision。迁移为one-sided local cell，front zero、tangent GT extent、behind slab分权；
- resolution：关闭radial ball，不调radius multiplier；M15预测GT-supervised tangent radius，normal support固定只向后
  `0.10m`，用同一CSG field训练/部署；
- claim impact：compact field有FREE precision正证据但无coverage claim。下一可用ID=`V71-F20`。

下一可用编号：`V71-F20`。

