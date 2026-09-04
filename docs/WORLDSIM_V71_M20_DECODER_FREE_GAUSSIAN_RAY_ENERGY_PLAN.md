# WorldSim V7.1 M20：无解码器的 Gaussian ray energy

## 研究问题

M19证明联合训练本身不足：可学习query decoder能够补偿已经恶化的completed point surface。M20删除该补偿通路，
只回答“GT first-return能否直接约束completed Gaussian geometry”。

## 表示

从canonical M8初始化surface head。对immutable observed anchors与全部generated children构成的共享
actor-canonical集合 `S={(c_j,s_j)}`，直接定义

\[
e(x)=\log\sum_j\exp\left(-\frac{\|x-c_j\|_2^2}{2s_j^2}\right).
\]

children的`c_j/s_j`均来自M8 head并持续接收native-3D geometry loss；observed anchor center不可学习且使用冻结
`0.08m` support。沿Actor AABB ray固定32个metric bins计算`e(x_k)`并softmax，GT target仍是唯一LiDAR
first-return bin。没有MLP/attention/query latent/visibility head可以吸收物理梯度。

该energy借鉴[GaussianFormer-2, CVPR 2025](https://github.com/huang-yh/GaussianFormer)的occupied-region Gaussian
probability view与[GaussRender, ICCV 2025](https://github.com/valeoai/GaussRender)的3D supervision + projective
loss并用原则；但不声称复现其scene occupancy或image pipeline。

## 监督与部署

- geometry：M8原生symmetric set Chamfer、local point-to-plane/scale、逐帧等权GT coverage全程保留；
- physics：同一Gaussian energy的ray categorical CE + 固定expected-depth L1；
- optimization：两个GT任务只对surface head做PCGrad；
- deployment：显式surface仍为immutable anchors + all children；ray首交读取同一energy softmax CDF median；
- 禁止：UNKNOWN mask、surface filter、learned decoder、threshold/temperature/scale sweep、Actor deletion。

energy是共享actor-canonical 3D函数，不读取target ray direction作为feature；ray direction只决定在哪些3D坐标查询。
trajectory继续是只读canonical-to-world刚体authority，static world独立；image/semantic/time/velocity/hazard不输入。

## 单次实验

- task=`WS-V71-M20-DECODER-FREE-GAUSSIAN-RAY-ENERGY-01`；seed=`71122`；M8初始化；
- 593 train Actors，固定4轮、32 bins、anchor scale `0.08m`；不做任何sweep；
- development 66 Actors有历史暴露，只作机制筛选；
- 最小判定：Actor/hazard retention 100%；point hazard early相对baseline至少降低5%；Chamfer相对M8恶化不超过
  1mm；energy绝对hazard early不高于冻结M18；energy all hit不低于冻结M18超过1pp；
- 若失败登记`V71-F24`并关闭isotropic Gaussian energy；若通过，在未读AV2 partial quality的前提下冻结同一
  20-log external evaluator。

