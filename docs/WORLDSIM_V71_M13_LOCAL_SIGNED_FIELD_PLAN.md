# WorldSim V7.1 M13 — M8-Guided Local Signed Field Plan

状态：`frozen / development-only`  
日期：2026-09-04  
分支：`research/worldsim-v7.1-learned-evidential-surface`

## 1. 高层纠偏

M9--M12依次排除了isotropic coupling、orientation、train/deploy renderer mismatch和normal thickness：所有独立primitive
union都无法同时改善earliest FREE collision与surface hit。M3/M4的implicit field也曾失败，但其条件是单一Actor global
latent，M4虽形成zero level却定位到错误表面。两条证据合并后的合理缺口是query-local identifiability，而非继续改scale。

M13假设：以已经通过GT point/frame监督的M8 children作为显式iso-points，让每个空间query读取nearest local child
features并只学习零初始化signed residual，可以同时保持已知表面位置和表达跨primitive连续可见边界。

## 2. 顶会依据

- ARO-Net（CVPR 2023）：https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_ARO-Net_Learning_Implicit_Fields_From_Anchored_Radial_Observations_CVPR_2023_paper.pdf
- Local Implicit Grid（CVPR 2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Jiang_Local_Implicit_Grid_Representations_for_3D_Scenes_CVPR_2020_paper.html
- IF-Net（CVPR 2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Chibane_Implicit_Functions_in_Feature_Space_for_3D_Shape_Reconstruction_and_CVPR_2020_paper.html
- LDIF（CVPR 2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Genova_Local_Deep_Implicit_Functions_for_3D_Shape_CVPR_2020_paper.html
- Iso-Points（CVPR 2021）：https://openaccess.thecvf.com/content/CVPR2021/papers/Yifan_Iso-Points_Optimizing_Neural_Implicit_Surfaces_With_Hybrid_Representations_CVPR_2021_paper.pdf

迁移query-specific local conditioning和explicit-point guidance；不迁移watertight ShapeNet occupancy prior、mesh GT或全局
test-time optimization。

## 3. 表示

对每个M8 child `i`保留冻结`(c_i,s_i)`，由其parent build feature、branch slot生成latent `z_i`。build PCA normal按
candidate-to-sensor ray定向到观测外侧，形成初始化local plane
`d_i(x)=(x-c_i)·n_i`。query取normalized Euclidean最近4个children，按距离softmax融合
`d_i(x)+0.25 s_i tanh(r_theta(z_i,(x-c_i)/s_i,d_i/s_i))`。residual末层全零，因此初始zero surface经过M8
children；不同于M3/M4，query永远保留局部identity/relative geometry。

## 4. GT supervision

每条train target ray `o→t`只定义窄带：

- `t-0.20d, t-0.10d`：FREE，signed target为正；
- `t`：surface zero；
- `t+0.05d, t+0.10d`：局部occupied，signed target为负；
- hit query的field gradient与GT endpoint 8NN normal作sign-invariant alignment。

后方超过`0.10m`区域保持unknown；不把整个cuboid interior伪标occupied。geometry（signed regression/hit/normal）与
physics（front-positive/back-negative）使用PCGrad。

## 5. 部署与评价

给定sensor ray与Actor AABB，以固定64 samples覆盖ray-box interval，查询同一signed field并线性插值首个
`positive→non-positive` crossing。零阈值由GT hit监督定义，不增加temperature/UNKNOWN/opacity threshold或后处理filter。

Development gates：

1. frozen M8 point五项合同保持；
2. field hazardous early相对M8 point first-return下降`>=5%`；
3. field all hit delta相对M8 point`>=-1pp`；
4. Actor/hazard retention=`100%`，所有local anchors保留。

单seed=`71115`、6 epochs、one run；不扫neighbor count、sample count、temperature、loss、seed、epoch。失败登记
`V71-F18`并关闭该local-field形式；通过才冻结fresh AV2 evaluator。即使通过，当前claim只覆盖observed-ray surface/free
space，不扩张为watertight interior或closed-loop safety。
